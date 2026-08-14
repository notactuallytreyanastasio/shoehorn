// The quant kernels are written index-for-index against ggml-quants.c so the
// port stays auditable against the reference; iterator style would hide that.
#![allow(clippy::needless_range_loop)]

mod fetch;
mod gguf;
mod imatrix;
mod iq_tables;
mod quant;
mod quant_iq;
mod solver;
mod vram;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use gguf::{GgmlType, Model, TensorInfo, Value};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::PathBuf;

fn progress(len: u64, msg: &'static str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{msg:>9} [{bar:30}] {pos}/{len} tensors ({elapsed})")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(msg);
    pb
}

#[derive(Parser)]
#[command(name = "shoehorn", about = "Quantize a BF16 GGUF with an imatrix to exactly fit your VRAM, then run it with llama.cpp")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct FitArgs {
    /// BF16 (or F16/F32) source GGUF
    #[arg(short, long)]
    model: PathBuf,
    /// imatrix file (legacy .imatrix or GGUF-based)
    #[arg(short, long)]
    imatrix: Option<PathBuf>,
    /// context length to budget the KV cache for
    #[arg(long, default_value_t = 8192)]
    ctx: u64,
    /// override detected VRAM, e.g. "18GiB", "800MB", or bytes
    #[arg(long)]
    budget: Option<String>,
    /// budget for a different Mac by its RAM size, e.g. "16GB"
    /// (approximates the macOS GPU working-set limit as 74% of RAM)
    #[arg(long, conflicts_with = "budget")]
    target: Option<String>,
    /// KV cache type to budget for and run with: f16, q8_0, or q4_0
    #[arg(long, default_value = "f16")]
    kv: String,
    /// safety margin subtracted from the budget
    /// (default 512MiB, or 160MiB with --calibrate)
    #[arg(long)]
    reserve: Option<String>,
    /// after writing, measure llama.cpp's real KV/compute allocations and
    /// re-solve with them, spending the recovered estimate slack on quality
    #[arg(long)]
    calibrate: bool,
    /// measure quantization error on all rows instead of a sample
    #[arg(long)]
    exact_errors: bool,
}

impl FitArgs {
    fn reserve_bytes(&self) -> Result<u64> {
        match &self.reserve {
            Some(s) => parse_size(s),
            None => Ok(if self.calibrate { 160 << 20 } else { 512 << 20 }),
        }
    }
}

fn kv_bytes_per_element(kv: &str) -> Result<f64> {
    Ok(match kv {
        "f16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q4_0" => 18.0 / 32.0,
        other => bail!("unsupported --kv {other} (use f16, q8_0, or q4_0)"),
    })
}

#[derive(Subcommand)]
enum Cmd {
    /// Show the solved per-tensor quant mix without writing anything
    Plan(FitArgs),
    /// Solve the mix and write the quantized GGUF
    Quantize {
        #[command(flatten)]
        fit: FitArgs,
        /// output GGUF path
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Launch llama-server on a model (extra args after -- go to llama-server)
    Run {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value_t = 8192)]
        ctx: u64,
        /// KV cache type: f16, q8_0, or q4_0
        #[arg(long, default_value = "f16")]
        kv: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// One-shot: fetch a model (path, HF owner/repo, or URL), get an imatrix,
    /// quantize to fit, and optionally serve it
    Fit {
        /// local GGUF path, Hugging Face repo id like unsloth/Qwen3-4B-GGUF, or URL
        model: String,
        #[arg(short, long)]
        imatrix: Option<PathBuf>,
        #[command(flatten)]
        fit: FitTuning,
        /// output GGUF path (default: <model-stem>-fit.gguf in the current dir)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// launch llama-server on the result
        #[arg(short, long)]
        serve: bool,
        /// extra args after -- go to llama-server with --serve
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// Print detected GPU memory
    Vram,
}

/// FitArgs minus the model/imatrix paths, for the `fit` subcommand.
#[derive(clap::Args, Clone)]
struct FitTuning {
    #[arg(long, default_value_t = 8192)]
    ctx: u64,
    #[arg(long)]
    budget: Option<String>,
    #[arg(long, conflicts_with = "budget")]
    target: Option<String>,
    #[arg(long, default_value = "f16")]
    kv: String,
    #[arg(long)]
    reserve: Option<String>,
    #[arg(long)]
    calibrate: bool,
    #[arg(long)]
    exact_errors: bool,
}

fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();
    let (num, mult) = if let Some(p) = lower.strip_suffix("gib") {
        (p, 1u64 << 30)
    } else if let Some(p) = lower.strip_suffix("mib") {
        (p, 1u64 << 20)
    } else if let Some(p) = lower.strip_suffix("gb") {
        (p, 1_000_000_000)
    } else if let Some(p) = lower.strip_suffix("mb") {
        (p, 1_000_000)
    } else if let Some(p) = lower.strip_suffix('g') {
        (p, 1u64 << 30)
    } else if let Some(p) = lower.strip_suffix('m') {
        (p, 1u64 << 20)
    } else {
        (lower.as_str(), 1)
    };
    let v: f64 = num.trim().parse().with_context(|| format!("bad size {s:?}"))?;
    Ok((v * mult as f64) as u64)
}

fn fmt_size(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    } else {
        format!("{b} B")
    }
}

struct Hyper {
    arch: String,
    n_layer: u64,
    n_kv_head: u64,
    key_len: u64,
    val_len: u64,
    n_embd: u64,
    n_vocab: u64,
}

fn hyperparams(f: &Model) -> Result<Hyper> {
    let arch = f
        .kv("general.architecture")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing general.architecture"))?
        .to_string();
    let get = |key: &str| -> Option<u64> { f.kv(&format!("{arch}.{key}")).and_then(Value::as_u64) };
    let n_layer = get("block_count").ok_or_else(|| anyhow!("missing block_count"))?;
    let n_embd = get("embedding_length").unwrap_or(0);
    let n_head = get("attention.head_count").unwrap_or(1).max(1);
    let n_kv_head = get("attention.head_count_kv").unwrap_or(n_head);
    let key_len = get("attention.key_length").unwrap_or(n_embd / n_head);
    let val_len = get("attention.value_length").unwrap_or(key_len);
    let n_vocab = get("vocab_size")
        .or_else(|| match f.kv("tokenizer.ggml.tokens") {
            Some(Value::Arr(_, v)) => Some(v.len() as u64),
            _ => None,
        })
        .unwrap_or(32000);
    Ok(Hyper { arch, n_layer, n_kv_head, key_len, val_len, n_embd, n_vocab })
}

struct Budget {
    usable_vram: u64,
    kv_bytes: u64,
    compute_bytes: u64,
    reserve: u64,
    model_budget: u64,
    source: String,
}

fn compute_budget(args: &FitArgs, h: &Hyper) -> Result<Budget> {
    let (usable_vram, source) = match (&args.budget, &args.target) {
        (Some(s), _) => (parse_size(s)?, format!("--budget {s}")),
        (None, Some(t)) => {
            let ram = parse_size(t)?;
            (ram * 74 / 100, format!("--target {t} (74% of RAM, macOS working-set approximation)"))
        }
        (None, None) => {
            let (b, name) = vram::probe().ok_or_else(|| anyhow!("no Metal device found; pass --budget"))?;
            (b, format!("Metal recommendedMaxWorkingSetSize ({name})"))
        }
    };
    let kv_elem = kv_bytes_per_element(&args.kv)?;
    let kv_bytes =
        (h.n_layer as f64 * args.ctx as f64 * h.n_kv_head as f64 * (h.key_len + h.val_len) as f64 * kv_elem) as u64;
    let ubatch = 512u64.min(args.ctx);
    let compute_bytes = ubatch * h.n_vocab * 4 + ubatch * h.n_embd * 4 * 8;
    let reserve = args.reserve_bytes()?;
    let overhead = kv_bytes + compute_bytes + reserve;
    if overhead >= usable_vram {
        bail!(
            "no room for weights: VRAM {} - KV {} - compute {} - reserve {} <= 0 (try smaller --ctx)",
            fmt_size(usable_vram), fmt_size(kv_bytes), fmt_size(compute_bytes), fmt_size(reserve)
        );
    }
    Ok(Budget {
        usable_vram,
        kv_bytes,
        compute_bytes,
        reserve,
        model_budget: usable_vram - overhead,
        source,
    })
}

/// What we plan to do with each tensor.
enum Disposition {
    /// copy/convert to this fixed type (norms, biases, non-weight tensors)
    Fixed(GgmlType),
    /// solver picks among candidates
    Solve(Vec<GgmlType>),
}

fn dispose(t: &TensorInfo) -> Disposition {
    let quantizable = t.dims.len() >= 2 && t.name.ends_with(".weight") && t.ne0().is_multiple_of(32);
    if !quantizable {
        // keep F32 for small/sensitive tensors (llama.cpp convention)
        let ty = match t.ty {
            GgmlType::F16 | GgmlType::Bf16 | GgmlType::F32 => GgmlType::F32,
            other => other, // already-quantized oddballs: copy untouched
        };
        return Disposition::Fixed(ty);
    }
    // Embeddings and the LM head crater below ~4 bpw in ways weighted MSE
    // understates (token_embd has no imatrix at all), so like llama.cpp's own
    // IQ2 mixes we floor these two at 4-bit.
    let sensitive = t.name == "token_embd.weight" || t.name == "output.weight";
    let mut c = if t.ne0().is_multiple_of(256) {
        let mut v = vec![GgmlType::Iq4Xs, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Q8_0];
        if !sensitive {
            v.extend([
                GgmlType::Iq2Xxs,
                GgmlType::Iq2Xs,
                GgmlType::Iq2S,
                GgmlType::Iq3Xxs,
                GgmlType::Iq3S,
            ]);
        }
        v
    } else {
        vec![GgmlType::Iq4Nl, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0]
    };
    c.push(GgmlType::F16);
    c
        .sort_by_key(|ty| ty.row_bytes(t.ne0()));
    Disposition::Solve(c)
}

/// Iterate a tensor's rows as f32, calling `f(row_idx, &row, imatrix_slice)`.
fn for_rows<F: FnMut(usize, &[f32], Option<&[f32]>)>(
    file: &Model,
    t: &TensorInfo,
    im: Option<&[f32]>,
    step: usize,
    mut f: F,
) -> Result<()> {
    let ne0 = t.ne0() as usize;
    let n_rows = t.n_rows() as usize;
    let rows_per_mat = if t.dims.len() >= 3 {
        t.dims[1] as usize
    } else {
        n_rows
    };
    let data = file.tensor_data(t);
    let src_row_bytes = t.ty.row_bytes(t.ne0()) as usize;
    let mut row = vec![0f32; ne0];
    let mut r = 0usize;
    while r < n_rows {
        let rd = &data[r * src_row_bytes..(r + 1) * src_row_bytes];
        row.clear();
        quant::decode_row(t.ty, rd, ne0, &mut row);
        let ims = imatrix::row_slice(im, ne0, rows_per_mat, r);
        f(r, &row, ims);
        r += step;
    }
    Ok(())
}

struct PlanEntry {
    tensor_idx: usize,
    ty: GgmlType,
    bytes: u64,
    solved: bool,
}

struct Plan {
    entries: Vec<PlanEntry>,
    budget: Budget,
    total_bytes: u64,
    /// cached measurement results, for cheap re-solves (--calibrate)
    choices: Vec<solver::TensorChoices>,
    fixed: Vec<(usize, GgmlType)>,
    fixed_bytes: u64,
}

fn assemble_entries(
    file: &Model,
    fixed: &[(usize, GgmlType)],
    choices: &[solver::TensorChoices],
    sel: &[usize],
) -> (Vec<PlanEntry>, u64) {
    let mut entries: Vec<PlanEntry> = fixed
        .iter()
        .map(|&(i, ty)| PlanEntry {
            tensor_idx: i,
            ty,
            bytes: ty.row_bytes(file.tensors[i].ne0()) * file.tensors[i].n_rows(),
            solved: false,
        })
        .collect();
    let mut solved_bytes = 0u64;
    for (tc, &ci) in choices.iter().zip(sel) {
        let c = &tc.cands[ci];
        solved_bytes += c.bytes;
        entries.push(PlanEntry { tensor_idx: tc.tensor_idx, ty: c.ty, bytes: c.bytes, solved: true });
    }
    entries.sort_by_key(|e| e.tensor_idx);
    (entries, solved_bytes)
}

fn build_plan(args: &FitArgs, file: &Model) -> Result<Plan> {
    let h = hyperparams(file)?;
    let budget = compute_budget(args, &h)?;
    let im = match &args.imatrix {
        Some(p) => imatrix::load(p.to_str().unwrap())?,
        None => {
            eprintln!("warning: no imatrix given; quantizing with activation-agnostic weights");
            imatrix::Imatrix::new()
        }
    };

    eprintln!(
        "model: {} | {} tensors | arch {} | ctx {}",
        fmt_size(file.total_bytes()),
        file.tensors.len(),
        h.arch,
        args.ctx
    );
    eprintln!(
        "budget: {} usable ({}) - {} KV - {} compute est - {} reserve = {} for weights",
        fmt_size(budget.usable_vram),
        budget.source,
        fmt_size(budget.kv_bytes),
        fmt_size(budget.compute_bytes),
        fmt_size(budget.reserve),
        fmt_size(budget.model_budget)
    );

    let mut fixed_bytes = 0u64;
    let mut fixed: Vec<(usize, GgmlType)> = Vec::new();
    let mut to_solve: Vec<(usize, Vec<GgmlType>)> = Vec::new();
    let mut n_with_imatrix = 0usize;
    for (i, t) in file.tensors.iter().enumerate() {
        match dispose(t) {
            Disposition::Fixed(ty) => {
                fixed_bytes += ty.row_bytes(t.ne0()) * t.n_rows();
                fixed.push((i, ty));
            }
            Disposition::Solve(c) => {
                if im.contains_key(t.name.trim_end_matches(".weight"))
                    || im.contains_key(&t.name)
                {
                    n_with_imatrix += 1;
                }
                to_solve.push((i, c));
            }
        }
    }
    eprintln!(
        "tensors: {} to solve ({} with imatrix data), {} fixed ({})",
        to_solve.len(),
        n_with_imatrix,
        fixed.len(),
        fmt_size(fixed_bytes)
    );
    if fixed_bytes >= budget.model_budget {
        bail!("fixed tensors alone exceed the weight budget");
    }

    // Measure weighted error for every (tensor, candidate).
    let sample_rows = if args.exact_errors { usize::MAX } else { 128 };
    let pb = progress(to_solve.len() as u64, "measuring");
    let choices: Vec<solver::TensorChoices> = to_solve
        .par_iter()
        .progress_with(pb)
        .map(|(idx, cands)| {
            let t = &file.tensors[*idx];
            let tim = im
                .get(&t.name)
                .or_else(|| im.get(t.name.trim_end_matches(".weight")))
                .map(|v| v.as_slice());
            let n_rows = t.n_rows() as usize;
            let step = (n_rows / sample_rows.min(n_rows)).max(1);
            let sampled = n_rows.div_ceil(step);
            let scale = n_rows as f64 / sampled as f64;
            let cands: Vec<solver::Candidate> = cands
                .par_iter()
                .map(|&ty| {
                    let mut err = 0f64;
                    for_rows(file, t, tim, step, |_r, row, ims| {
                        err += quant::row_error(ty, row, ims);
                    })
                    .unwrap();
                    solver::Candidate {
                        ty,
                        bytes: ty.row_bytes(t.ne0()) * t.n_rows(),
                        err: err * scale,
                    }
                })
                .collect();
            solver::TensorChoices { tensor_idx: *idx, cands }
        })
        .collect();

    let weight_budget = budget.model_budget - fixed_bytes;
    let sel = solver::solve(&choices, weight_budget).ok_or_else(|| {
        let min: u64 = choices.iter().map(|t| t.cands[0].bytes).sum();
        anyhow!(
            "even the smallest mix ({} + {} fixed) exceeds the weight budget {}; \
             lower --ctx / --reserve or use a smaller model",
            fmt_size(min),
            fmt_size(fixed_bytes),
            fmt_size(budget.model_budget)
        )
    })?;

    let (entries, solved_bytes) = assemble_entries(file, &fixed, &choices, &sel);
    Ok(Plan {
        entries,
        budget,
        total_bytes: fixed_bytes + solved_bytes,
        choices,
        fixed,
        fixed_bytes,
    })
}

fn print_plan(plan: &Plan, file: &Model) {
    println!("\n{:<42} {:>16} {:>7} {:>12} {:>6}", "tensor", "shape", "type", "size", "bpw");
    let mut by_type: std::collections::BTreeMap<String, (usize, u64)> = Default::default();
    for e in &plan.entries {
        let t = &file.tensors[e.tensor_idx];
        let shape = t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
        let bpw = e.bytes as f64 * 8.0 / t.n_elements() as f64;
        let ent = by_type.entry(e.ty.name()).or_default();
        ent.0 += 1;
        ent.1 += e.bytes;
        println!("{:<42} {:>16} {:>7} {:>12} {:>6.2}", t.name, shape, e.ty.name(), fmt_size(e.bytes), bpw);
    }
    println!("\nby type:");
    for (ty, (n, bytes)) in &by_type {
        println!("  {:>6}: {:>3} tensors, {}", ty, n, fmt_size(*bytes));
    }
    let b = &plan.budget;
    let total_elems: u64 = file.tensors.iter().map(|t| t.n_elements()).sum();
    println!(
        "\nweights: {} of {} budget ({:.3}% used, {} slack) | overall {:.3} bpw",
        fmt_size(plan.total_bytes),
        fmt_size(b.model_budget),
        100.0 * plan.total_bytes as f64 / b.model_budget as f64,
        fmt_size(b.model_budget - plan.total_bytes),
        plan.total_bytes as f64 * 8.0 / total_elems as f64,
    );
    println!(
        "projected VRAM at ctx: {} weights + {} KV + {} compute + {} reserve = {} of {} available",
        fmt_size(plan.total_bytes),
        fmt_size(b.kv_bytes),
        fmt_size(b.compute_bytes),
        fmt_size(b.reserve),
        fmt_size(plan.total_bytes + b.kv_bytes + b.compute_bytes + b.reserve),
        fmt_size(b.usable_vram)
    );
}

/// GGUF general.file_type value for the dominant quantized type (cosmetic).
fn file_type_code(entries: &[PlanEntry]) -> u32 {
    let mut best = (0u64, 1u32);
    for e in entries {
        if !e.solved {
            continue;
        }
        let code = match e.ty {
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 8,
            GgmlType::Q5_1 => 9,
            GgmlType::Q8_0 => 7,
            GgmlType::Q4K => 15,
            GgmlType::Q5K => 17,
            GgmlType::Q6K => 18,
            GgmlType::Iq2Xxs => 19,
            GgmlType::Iq2Xs => 20,
            GgmlType::Iq3Xxs => 23,
            GgmlType::Iq4Nl => 25,
            GgmlType::Iq3S => 26,
            GgmlType::Iq2S => 28,
            GgmlType::Iq4Xs => 30,
            _ => 1,
        };
        if e.bytes > best.0 {
            best = (e.bytes, code);
        }
    }
    best.1
}

/// Re-encode one tensor to `ty`, parallelizing across row chunks so a single
/// huge tensor (token_embd) doesn't serialize the encode.
fn encode_tensor(file: &Model, t: &TensorInfo, ty: GgmlType, im: Option<&[f32]>) -> Vec<u8> {
    let ne0 = t.ne0() as usize;
    let n_rows = t.n_rows() as usize;
    let rows_per_mat = if t.dims.len() >= 3 { t.dims[1] as usize } else { n_rows };
    let src = file.tensor_data(t);
    let src_rb = t.ty.row_bytes(t.ne0()) as usize;
    let dst_rb = ty.row_bytes(t.ne0()) as usize;
    let mut out = vec![0u8; n_rows * dst_rb];
    let chunk_rows = (n_rows / (rayon::current_num_threads() * 4)).clamp(1, 512);
    out.par_chunks_mut(chunk_rows * dst_rb).enumerate().for_each(|(ci, dst)| {
        let r0 = ci * chunk_rows;
        let mut row = Vec::with_capacity(ne0);
        let mut enc = Vec::with_capacity(dst_rb);
        for (k, drow) in dst.chunks_mut(dst_rb).enumerate() {
            let r = r0 + k;
            row.clear();
            quant::decode_row(t.ty, &src[r * src_rb..(r + 1) * src_rb], ne0, &mut row);
            enc.clear();
            quant::encode_row(ty, &row, imatrix::row_slice(im, ne0, rows_per_mat, r), &mut enc);
            drow.copy_from_slice(&enc);
        }
    });
    out
}

/// Streaming write: one tensor in memory at a time. When `reuse` is given
/// (a previously written output plus its plan), tensors whose type is
/// unchanged are copied raw from it instead of re-encoded.
fn write_quantized(
    out_path: &PathBuf,
    plan: &Plan,
    file: &Model,
    im: &imatrix::Imatrix,
    reuse: Option<(&Model, &Plan)>,
) -> Result<()> {
    let infos: Vec<TensorInfo> = plan
        .entries
        .iter()
        .map(|e| {
            let t = &file.tensors[e.tensor_idx];
            TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty: e.ty, offset: 0, shard: 0 }
        })
        .collect();

    // Drop split.* markers: the output is a single merged file, and keeping
    // split.count would send llama.cpp looking for sibling shards.
    let mut kvs: Vec<(String, Value)> = file
        .kvs
        .iter()
        .filter(|(k, _)| {
            k != "general.file_type" && k != "general.quantization_version" && !k.starts_with("split.")
        })
        .cloned()
        .collect();
    kvs.push(("general.file_type".into(), Value::U32(file_type_code(&plan.entries))));
    kvs.push(("general.quantization_version".into(), Value::U32(2)));

    let pb = progress(plan.entries.len() as u64, "encoding");
    let mut reused = 0usize;
    let tmp_path = out_path.with_extension("gguf.tmp");
    let out = std::fs::File::create(&tmp_path)
        .with_context(|| format!("creating {}", tmp_path.display()))?;
    let written = gguf::write_streaming(out, &kvs, &infos, file.alignment, |i| {
        let e = &plan.entries[i];
        let t = &file.tensors[e.tensor_idx];
        pb.inc(1);
        if let Some((old, old_plan)) = reuse
            && old_plan.entries[i].ty == e.ty
        {
            reused += 1;
            return Ok(old.tensor_data(&old.tensors[i]).to_vec());
        }
        if e.ty == t.ty {
            return Ok(file.tensor_data(t).to_vec());
        }
        let tim = im
            .get(&t.name)
            .or_else(|| im.get(t.name.trim_end_matches(".weight")))
            .map(|v| v.as_slice());
        Ok(encode_tensor(file, t, e.ty, tim))
    })?;
    pb.finish_and_clear();
    std::fs::rename(&tmp_path, out_path)?;
    if reused > 0 {
        eprintln!("reused {reused} unchanged tensors from the previous pass");
    }
    eprintln!("wrote {} ({})", out_path.display(), fmt_size(written));
    Ok(())
}

/// Run llama.cpp briefly on a model and read back its real KV-cache and
/// compute-buffer allocations from the verbose log.
fn measure_runtime(model: &PathBuf, ctx: u64, kv: &str) -> Result<(u64, u64)> {
    eprintln!("calibrating: loading the model in llama.cpp to measure real allocations ...");
    let mut cmd = std::process::Command::new("llama-cli");
    cmd.arg("-m")
        .arg(model)
        .args(["-ngl", "99", "-c", &ctx.to_string(), "-n", "1", "--temp", "0", "-st", "-v", "-p", "hi"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null());
    if kv != "f16" {
        cmd.args(["--cache-type-k", kv, "--cache-type-v", kv]);
    }
    let out = cmd.output().context("running llama-cli for calibration")?;
    let log = String::from_utf8_lossy(&out.stderr);
    let mib = |line: &str| -> Option<f64> {
        let (_, rest) = line.split_once("size =")?;
        rest.trim().strip_suffix("MiB").map(|n| n.trim().parse().ok())?
    };
    let mut kv_bytes = None;
    let mut compute = 0f64;
    for line in log.lines() {
        if line.contains("llama_kv_cache: size =") {
            kv_bytes = line.split_once('(').and_then(|(head, _)| mib(head));
        } else if line.contains("compute buffer size =") {
            compute += mib(line).unwrap_or(0.0);
        }
    }
    let kv_bytes = kv_bytes.ok_or_else(|| {
        anyhow!("could not find KV cache size in llama-cli output (is the model loadable?)")
    })?;
    if compute == 0.0 {
        bail!("could not find compute buffer sizes in llama-cli output");
    }
    Ok(((kv_bytes * 1048576.0) as u64, (compute * 1048576.0) as u64))
}

/// The --calibrate pass: measure real allocations, re-solve with them, and
/// rewrite the output reusing every unchanged tensor.
fn calibrate_pass(
    out_path: &PathBuf,
    plan: &Plan,
    args: &FitArgs,
    file: &Model,
    im: &imatrix::Imatrix,
) -> Result<()> {
    let (kv_meas, compute_meas) = measure_runtime(out_path, args.ctx, &args.kv)?;
    let reserve = args.reserve_bytes()?;
    let est_overhead = plan.budget.kv_bytes + plan.budget.compute_bytes;
    let meas_overhead = kv_meas + compute_meas;
    eprintln!(
        "measured: {} KV + {} compute = {} (estimate was {}; {} reclaimed)",
        fmt_size(kv_meas),
        fmt_size(compute_meas),
        fmt_size(meas_overhead),
        fmt_size(est_overhead),
        if est_overhead > meas_overhead {
            fmt_size(est_overhead - meas_overhead)
        } else {
            format!("-{}", fmt_size(meas_overhead - est_overhead))
        },
    );
    if meas_overhead + reserve + plan.fixed_bytes >= plan.budget.usable_vram {
        bail!("measured overhead exceeds the memory envelope; lower --ctx");
    }
    let new_weight_budget = plan.budget.usable_vram - meas_overhead - reserve - plan.fixed_bytes;
    let sel = solver::solve(&plan.choices, new_weight_budget)
        .ok_or_else(|| anyhow!("re-solve infeasible (this should not happen)"))?;
    let (entries, solved_bytes) = assemble_entries(file, &plan.fixed, &plan.choices, &sel);
    let changed = entries
        .iter()
        .zip(&plan.entries)
        .filter(|(a, b)| a.ty != b.ty)
        .count();
    let new_plan = Plan {
        entries,
        budget: Budget {
            kv_bytes: kv_meas,
            compute_bytes: compute_meas,
            reserve,
            model_budget: new_weight_budget + plan.fixed_bytes,
            usable_vram: plan.budget.usable_vram,
            source: format!("{} + calibration", plan.budget.source),
        },
        total_bytes: plan.fixed_bytes + solved_bytes,
        choices: Vec::new(),
        fixed: plan.fixed.clone(),
        fixed_bytes: plan.fixed_bytes,
    };
    if changed == 0 {
        eprintln!("calibration changed no tensor choices; keeping the first pass");
        return Ok(());
    }
    eprintln!(
        "calibration upgrades {} tensors ({} -> {} weights)",
        changed,
        fmt_size(plan.total_bytes),
        fmt_size(new_plan.total_bytes)
    );
    let old = Model::open(out_path)?;
    let calibrated = out_path.clone();
    write_quantized(&calibrated, &new_plan, file, im, Some((&old, plan)))?;
    print_plan(&new_plan, file);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Vram => {
            match vram::probe() {
                Some((b, name)) => println!("{name}: {} usable for GPU working set", fmt_size(b)),
                None => println!("no Metal device found"),
            }
            Ok(())
        }
        Cmd::Plan(args) => {
            let file = Model::open(&args.model)?;
            let plan = build_plan(&args, &file)?;
            print_plan(&plan, &file);
            Ok(())
        }
        Cmd::Quantize { fit, output } => {
            let file = Model::open(&fit.model)?;
            let plan = build_plan(&fit, &file)?;
            print_plan(&plan, &file);
            let im = match &fit.imatrix {
                Some(p) => imatrix::load(p.to_str().unwrap())?,
                None => imatrix::Imatrix::new(),
            };
            write_quantized(&output, &plan, &file, &im, None)?;
            if fit.calibrate {
                calibrate_pass(&output, &plan, &fit, &file, &im)?;
            }
            Ok(())
        }
        Cmd::Run { model, ctx, kv, extra } => serve(&model, ctx, &kv, &extra),
        Cmd::Fit { model, imatrix, fit, output, serve: do_serve, extra } => {
            let resolved = fetch::resolve(&model)?;
            let imatrix_path = match imatrix {
                Some(p) => Some(p),
                None => match resolved.imatrix {
                    Some(p) => Some(p),
                    None => fetch::auto_imatrix(&resolved.model, vram::probe().map(|(b, _)| b))?,
                },
            };
            let args = FitArgs {
                model: resolved.model.clone(),
                imatrix: imatrix_path,
                ctx: fit.ctx,
                budget: fit.budget,
                target: fit.target,
                kv: fit.kv.clone(),
                reserve: fit.reserve,
                calibrate: fit.calibrate,
                exact_errors: fit.exact_errors,
            };
            let out_path = output.unwrap_or_else(|| {
                let stem = resolved
                    .model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".into());
                PathBuf::from(format!("{stem}-fit.gguf"))
            });
            let file = Model::open(&args.model)?;
            let plan = build_plan(&args, &file)?;
            print_plan(&plan, &file);
            let im = match &args.imatrix {
                Some(p) => imatrix::load(p.to_str().unwrap())?,
                None => imatrix::Imatrix::new(),
            };
            write_quantized(&out_path, &plan, &file, &im, None)?;
            if args.calibrate {
                calibrate_pass(&out_path, &plan, &args, &file, &im)?;
            }
            if do_serve {
                serve(&out_path, args.ctx, &args.kv, &extra)
            } else {
                eprintln!(
                    "\nrun it: shoehorn run -m {} --ctx {}{}",
                    out_path.display(),
                    args.ctx,
                    if args.kv != "f16" { format!(" --kv {}", args.kv) } else { String::new() }
                );
                Ok(())
            }
        }
    }
}

/// exec llama-server on a model with full offload and the budgeted KV type.
fn serve(model: &PathBuf, ctx: u64, kv: &str, extra: &[String]) -> Result<()> {
    use std::os::unix::process::CommandExt;
    kv_bytes_per_element(kv)?; // validate
    let mut cmd = std::process::Command::new("llama-server");
    cmd.arg("-m").arg(model).arg("-c").arg(ctx.to_string()).arg("-ngl").arg("99");
    if kv != "f16" {
        cmd.args(["--cache-type-k", kv, "--cache-type-v", kv]);
    }
    cmd.args(extra);
    eprintln!("exec: {cmd:?}");
    Err(cmd.exec().into())
}
