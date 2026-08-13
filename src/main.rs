mod gguf;
mod imatrix;
mod quant;
mod solver;
mod vram;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use gguf::{GgmlType, GgufFile, TensorInfo, Value};
use rayon::prelude::*;
use std::path::PathBuf;

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
    /// safety margin subtracted from the budget
    #[arg(long, default_value = "512MiB")]
    reserve: String,
    /// measure quantization error on all rows instead of a sample
    #[arg(long)]
    exact_errors: bool,
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
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        extra: Vec<String>,
    },
    /// Print detected GPU memory
    Vram,
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

fn hyperparams(f: &GgufFile) -> Result<Hyper> {
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
    let (usable_vram, source) = match &args.budget {
        Some(s) => (parse_size(s)?, format!("--budget {s}")),
        None => {
            let (b, name) = vram::probe().ok_or_else(|| anyhow!("no Metal device found; pass --budget"))?;
            (b, format!("Metal recommendedMaxWorkingSetSize ({name})"))
        }
    };
    let kv_bytes = h.n_layer * args.ctx * h.n_kv_head * (h.key_len + h.val_len) * 2;
    let ubatch = 512u64.min(args.ctx);
    let compute_bytes = ubatch * h.n_vocab * 4 + ubatch * h.n_embd * 4 * 8;
    let reserve = parse_size(&args.reserve)?;
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
    let quantizable = t.dims.len() >= 2 && t.name.ends_with(".weight") && t.ne0() % 32 == 0;
    if !quantizable {
        // keep F32 for small/sensitive tensors (llama.cpp convention)
        let ty = match t.ty {
            GgmlType::F16 | GgmlType::Bf16 | GgmlType::F32 => GgmlType::F32,
            other => other, // already-quantized oddballs: copy untouched
        };
        return Disposition::Fixed(ty);
    }
    let mut c = if t.ne0() % 256 == 0 {
        vec![GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K, GgmlType::Q8_0]
    } else {
        vec![GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0]
    };
    c.push(GgmlType::F16);
    c
        .sort_by_key(|ty| ty.row_bytes(t.ne0()));
    Disposition::Solve(c)
}

/// Iterate a tensor's rows as f32, calling `f(row_idx, &row, imatrix_slice)`.
fn for_rows<F: FnMut(usize, &[f32], Option<&[f32]>)>(
    file: &GgufFile,
    buf: &[u8],
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
    let data = file.tensor_data(buf, t);
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
}

fn build_plan(args: &FitArgs, file: &GgufFile, buf: &[u8]) -> Result<Plan> {
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
        fmt_size(buf.len() as u64),
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
    eprintln!("measuring quantization error for each tensor x candidate ...");
    let choices: Vec<solver::TensorChoices> = to_solve
        .par_iter()
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
                    for_rows(file, buf, t, tim, step, |_r, row, ims| {
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

    let mut entries: Vec<PlanEntry> = fixed
        .iter()
        .map(|&(i, ty)| PlanEntry {
            tensor_idx: i,
            ty,
            bytes: ty.row_bytes(file.tensors[i].ne0()) * file.tensors[i].n_rows(),
            solved: false,
        })
        .collect();
    for (tc, &ci) in choices.iter().zip(&sel) {
        let c = &tc.cands[ci];
        entries.push(PlanEntry { tensor_idx: tc.tensor_idx, ty: c.ty, bytes: c.bytes, solved: true });
    }
    entries.sort_by_key(|e| e.tensor_idx);
    let total_bytes = fixed_bytes + choices.iter().zip(&sel).map(|(t, &c)| t.cands[c].bytes).sum::<u64>();
    Ok(Plan { entries, budget, total_bytes })
}

fn print_plan(plan: &Plan, file: &GgufFile) {
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
            _ => 1,
        };
        if e.bytes > best.0 {
            best = (e.bytes, code);
        }
    }
    best.1
}

fn write_quantized(
    out_path: &PathBuf,
    plan: &Plan,
    file: &GgufFile,
    buf: &[u8],
    im: &imatrix::Imatrix,
) -> Result<()> {
    eprintln!("encoding {} tensors ...", plan.entries.len());
    let tensors: Vec<(TensorInfo, Vec<u8>)> = plan
        .entries
        .par_iter()
        .map(|e| {
            let t = &file.tensors[e.tensor_idx];
            let tim = im
                .get(&t.name)
                .or_else(|| im.get(t.name.trim_end_matches(".weight")))
                .map(|v| v.as_slice());
            let mut data = Vec::with_capacity(e.bytes as usize);
            if e.ty == t.ty {
                data.extend_from_slice(file.tensor_data(buf, t));
            } else {
                for_rows(file, buf, t, tim, 1, |_r, row, ims| {
                    quant::encode_row(e.ty, row, ims, &mut data);
                })
                .unwrap();
            }
            let info = TensorInfo { name: t.name.clone(), dims: t.dims.clone(), ty: e.ty, offset: 0 };
            (info, data)
        })
        .collect();

    let mut kvs: Vec<(String, Value)> = file
        .kvs
        .iter()
        .filter(|(k, _)| k != "general.file_type" && k != "general.quantization_version")
        .cloned()
        .collect();
    kvs.push(("general.file_type".into(), Value::U32(file_type_code(&plan.entries))));
    kvs.push(("general.quantization_version".into(), Value::U32(2)));

    let out = std::fs::File::create(out_path)
        .with_context(|| format!("creating {}", out_path.display()))?;
    let written = gguf::write(out, &kvs, &tensors, file.alignment)?;
    eprintln!("wrote {} ({})", out_path.display(), fmt_size(written));
    Ok(())
}

fn load_model(path: &PathBuf) -> Result<(memmap2::Mmap, GgufFile)> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let map = unsafe { memmap2::Mmap::map(&f)? };
    let g = gguf::read(&map)?;
    Ok((map, g))
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
            let (map, file) = load_model(&args.model)?;
            let plan = build_plan(&args, &file, &map)?;
            print_plan(&plan, &file);
            Ok(())
        }
        Cmd::Quantize { fit, output } => {
            let (map, file) = load_model(&fit.model)?;
            let plan = build_plan(&fit, &file, &map)?;
            print_plan(&plan, &file);
            let im = match &fit.imatrix {
                Some(p) => imatrix::load(p.to_str().unwrap())?,
                None => imatrix::Imatrix::new(),
            };
            write_quantized(&output, &plan, &file, &map, &im)?;
            Ok(())
        }
        Cmd::Run { model, ctx, extra } => {
            use std::os::unix::process::CommandExt;
            let mut cmd = std::process::Command::new("llama-server");
            cmd.arg("-m").arg(&model).arg("-c").arg(ctx.to_string()).arg("-ngl").arg("99");
            cmd.args(&extra);
            eprintln!("exec: {cmd:?}");
            Err(cmd.exec().into())
        }
    }
}
