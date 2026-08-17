//! End-to-end pipeline test: synthesize a tiny BF16 GGUF, fit it to a tight
//! budget, and re-read the output. Exercises the reader, budget model,
//! measurement, solver, encoders, and streaming writer together — the parts
//! unit tests cover only in isolation.

use crate::gguf::{GgmlType, Model, TensorInfo, Value};
use crate::{build_plan, quant, write_quantized, FitArgs};
use std::path::PathBuf;

/// Deterministic pseudo-random weights so the test never flakes.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

fn bf16_rows(seed: u64, ne0: usize, n_rows: usize) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(ne0 * n_rows * 2);
    for _ in 0..ne0 * n_rows {
        out.extend_from_slice(&half::bf16::from_f32(rng.next_f32()).to_bits().to_le_bytes());
    }
    out
}

fn synth_model(path: &PathBuf) {
    let kvs: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::Str("llama".into())),
        ("llama.block_count".into(), Value::U32(2)),
        ("llama.embedding_length".into(), Value::U32(256)),
        ("llama.attention.head_count".into(), Value::U32(4)),
        ("llama.attention.head_count_kv".into(), Value::U32(2)),
        ("llama.vocab_size".into(), Value::U32(256)),
    ];
    let spec: Vec<(&str, Vec<u64>, GgmlType)> = vec![
        ("token_embd.weight", vec![256, 128], GgmlType::Bf16),
        ("blk.0.attn_q.weight", vec![256, 256], GgmlType::Bf16),
        ("blk.0.ffn_down.weight", vec![256, 64], GgmlType::Bf16),
        ("blk.1.attn_q.weight", vec![256, 256], GgmlType::Bf16),
        ("output_norm.weight", vec![256], GgmlType::F32),
    ];
    let infos: Vec<TensorInfo> = spec
        .iter()
        .map(|(name, dims, ty)| TensorInfo {
            name: name.to_string(),
            dims: dims.clone(),
            ty: *ty,
            offset: 0,
            shard: 0,
        })
        .collect();
    let out = std::fs::File::create(path).unwrap();
    crate::gguf::write_streaming(out, &kvs, &infos, 32, |i| {
        let (_, dims, ty) = &spec[i];
        let n: usize = dims.iter().product::<u64>() as usize;
        Ok(match ty {
            GgmlType::Bf16 => bf16_rows(i as u64 + 1, dims[0] as usize, n / dims[0] as usize),
            _ => {
                let mut rng = Lcg(99);
                (0..n).flat_map(|_| rng.next_f32().to_le_bytes()).collect()
            }
        })
    })
    .unwrap();
}

#[test]
fn pipeline_fits_a_synthetic_model_into_a_tight_budget() {
    let dir = std::env::temp_dir().join(format!("shoehorn-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("synthetic-bf16.gguf");
    let out_path = dir.join("synthetic-fit.gguf");
    synth_model(&src_path);

    // Budget chosen so a pure-F16 mix (352 KiB of weights) cannot fit but a
    // mixed assignment can: overhead at ctx 64 is KV 64 KiB + compute 576 KiB,
    // leaving ~200 KiB for weights.
    let overhead = 64 * 1024 + 576 * 1024;
    let weight_room = 200 * 1024;
    let args = FitArgs {
        model: src_path.clone(),
        imatrix: None,
        ctx: 64,
        budget: Some((overhead + weight_room).to_string()),
        target: None,
        kv: "f16".into(),
        reserve: Some("0".into()),
        calibrate: false,
        exact_errors: true,
    };

    let file = Model::open(&src_path).unwrap();
    let plan = build_plan(&args, &file).unwrap();

    assert!(
        plan.total_bytes <= plan.budget.model_budget,
        "plan overshoots: {} > {}",
        plan.total_bytes,
        plan.budget.model_budget
    );
    assert!(
        plan.total_bytes as f64 >= plan.budget.model_budget as f64 * 0.85,
        "solver left too much on the table: {} of {}",
        plan.total_bytes,
        plan.budget.model_budget
    );
    let solved_types: std::collections::BTreeSet<_> =
        plan.entries.iter().filter(|e| e.solved).map(|e| e.ty.name()).collect();
    assert!(
        plan.entries.iter().any(|e| e.solved && e.ty != GgmlType::F16),
        "budget was tight but nothing got quantized (types: {solved_types:?})"
    );

    let im = crate::imatrix::Imatrix::new();
    write_quantized(&out_path, &plan, &file, &im, None).unwrap();

    let refit = Model::open(&out_path).unwrap();
    assert_eq!(refit.tensors.len(), file.tensors.len());
    assert_eq!(
        refit.kv("general.architecture").and_then(Value::as_str),
        Some("llama")
    );
    assert!(refit.kv("general.file_type").is_some());
    for (a, b) in file.tensors.iter().zip(&refit.tensors) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.dims, b.dims);
    }
    // Every re-read tensor decodes to finite values.
    for t in &refit.tensors {
        let ne0 = t.ne0() as usize;
        let mut row = Vec::with_capacity(ne0);
        quant::decode_row(t.ty, &refit.tensor_data(t)[..t.ty.row_bytes(t.ne0()) as usize], ne0, &mut row);
        assert!(row.iter().all(|v| v.is_finite()), "{} decodes to non-finite values", t.name);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
