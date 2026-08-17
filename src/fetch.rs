//! Model resolution for `shoehorn fit`: local path, Hugging Face repo id
//! (`owner/repo`), or direct URL. Downloads land in ~/.cache/shoehorn and are
//! resumed if interrupted.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Resolved {
    pub model: PathBuf,
    /// imatrix found alongside the model in its repo, if any
    pub imatrix: Option<PathBuf>,
}

pub fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    Ok(PathBuf::from(home).join(".cache/shoehorn"))
}

pub fn resolve(spec: &str) -> Result<Resolved> {
    let p = Path::new(spec);
    if p.exists() {
        return Ok(Resolved { model: p.to_path_buf(), imatrix: None });
    }
    if spec.starts_with("http://") || spec.starts_with("https://") {
        let name = spec
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("cannot infer filename from URL"))?;
        let dir = cache_dir()?.join("url");
        let model = download(spec, &dir, name)?;
        return Ok(Resolved { model, imatrix: None });
    }
    // owner/repo on Hugging Face
    let mut parts = spec.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(o), Some(r), None) if !o.is_empty() && !r.is_empty() => hf_repo(o, r),
        _ => bail!("model not found: {spec:?} is not a file, URL, or owner/repo id"),
    }
}

fn hf_repo(owner: &str, repo: &str) -> Result<Resolved> {
    let api = format!("https://huggingface.co/api/models/{owner}/{repo}/tree/main");
    let out = Command::new("curl")
        .args(["-sL", "--fail", &api])
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("Hugging Face API request failed for {owner}/{repo} (repo exists and is public?)");
    }
    let files: serde_json::Value = serde_json::from_slice(&out.stdout)
        .context("parsing Hugging Face API response")?;
    let entries: Vec<(String, u64)> = files
        .as_array()
        .ok_or_else(|| anyhow!("unexpected API response shape"))?
        .iter()
        .filter_map(|f| {
            Some((
                f.get("path")?.as_str()?.to_string(),
                f.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
            ))
        })
        .collect();

    let pick = |pred: &dyn Fn(&str) -> bool| -> Option<&(String, u64)> {
        entries
            .iter()
            .filter(|(p, _)| p.ends_with(".gguf") && pred(&p.to_lowercase()))
            .max_by_key(|(_, s)| *s)
    };
    let model_file = pick(&|p| p.contains("bf16"))
        .or_else(|| pick(&|p| p.contains("f16") && !p.contains("bf16")))
        .ok_or_else(|| {
            anyhow!(
                "no BF16/F16 GGUF in {owner}/{repo}; files: {}",
                entries
                    .iter()
                    .filter(|(p, _)| p.ends_with(".gguf"))
                    .map(|(p, _)| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    // A split model means downloading every sibling shard; Model::open then
    // reads them as one. The first shard is what we hand back.
    let stem = model_file.0.clone();
    let shard_files: Vec<&(String, u64)> = if let Some((prefix, _)) = stem.split_once("-of-") {
        let prefix = prefix.rsplit_once('-').map(|(p, _)| p).unwrap_or(&stem);
        let mut v: Vec<&(String, u64)> = entries
            .iter()
            .filter(|(p, _)| p.starts_with(prefix) && p.contains("-of-") && p.ends_with(".gguf"))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    } else {
        vec![model_file]
    };

    let dir = cache_dir()?.join(format!("{owner}__{repo}"));
    let total: u64 = shard_files.iter().map(|(_, s)| s).sum();
    eprintln!(
        "fetching {}/{} ({} file(s), {:.1} GB) ...",
        owner,
        shard_files[0].0,
        shard_files.len(),
        total as f64 / 1e9
    );
    let mut model = None;
    for (p, _) in &shard_files {
        let path = download(
            &format!("https://huggingface.co/{owner}/{repo}/resolve/main/{p}"),
            &dir,
            Path::new(p).file_name().unwrap().to_str().unwrap(),
        )?;
        model.get_or_insert(path);
    }
    let model = model.unwrap();

    let imatrix = entries
        .iter()
        .find(|(p, _)| p.to_lowercase().contains("imatrix"))
        .map(|(p, _)| {
            eprintln!("repo has an imatrix: {p}");
            download(
                &format!("https://huggingface.co/{owner}/{repo}/resolve/main/{p}"),
                &dir,
                p,
            )
        })
        .transpose()?;

    Ok(Resolved { model, imatrix })
}

/// curl with resume; progress goes to the user's terminal.
fn download(url: &str, dir: &Path, name: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(name);
    let part = dir.join(format!("{name}.part"));
    if dest.exists() {
        return Ok(dest);
    }
    let status = Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-C", "-", "--progress-bar", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl")?;
    if !status.success() {
        bail!("download failed: {url}");
    }
    std::fs::rename(&part, &dest)?;
    Ok(dest)
}

/// Held-out text for `shoehorn eval`: man pages disjoint from the
/// calibration set, so a model whose imatrix came from auto_imatrix isn't
/// evaluated on its own calibration data.
pub fn heldout_text() -> Result<PathBuf> {
    let path = cache_dir()?.join("heldout.txt");
    if path.exists() {
        return Ok(path);
    }
    let stdout = Command::new("sh")
        .args(["-c", "(man grep | col -b; man tar | col -b; man sed | col -b) 2>/dev/null"])
        .output()
        .map(|o| o.stdout)
        .unwrap_or_default();
    if stdout.len() < 50_000 {
        bail!("could not build held-out text from man pages; pass -f <textfile>");
    }
    std::fs::write(&path, &stdout)?;
    Ok(path)
}

/// Generate an imatrix with llama-imatrix using man-page calibration text.
/// Only attempted when the model comfortably fits the GPU working set.
pub fn auto_imatrix(model: &Path, vram: Option<u64>) -> Result<Option<PathBuf>> {
    if Command::new("llama-imatrix").arg("--version").output().is_err() {
        eprintln!("llama-imatrix not found; continuing without an imatrix");
        return Ok(None);
    }
    let model_bytes = std::fs::metadata(model)?.len();
    match vram {
        Some(v) if model_bytes < v * 4 / 5 => {}
        _ => {
            eprintln!(
                "model too large to generate an imatrix on this GPU; \
                 pass -i with a downloaded imatrix for better low-bit quality"
            );
            return Ok(None);
        }
    }
    let out = model.with_extension("imatrix.gguf");
    if out.exists() {
        return Ok(Some(out));
    }
    let calib = cache_dir()?.join("calibration.txt");
    if !calib.exists() {
        // Best-effort: no sh / no man pages (e.g. Windows, slim containers)
        // just means no auto-imatrix, not a failed fit.
        let stdout = Command::new("sh")
            .args(["-c", "(man bash | col -b; man zshexpn | col -b) 2>/dev/null"])
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default();
        if stdout.len() < 100_000 {
            eprintln!("could not build calibration text; continuing without an imatrix");
            return Ok(None);
        }
        std::fs::write(&calib, &stdout)?;
    }
    eprintln!("generating imatrix (one-time, a few minutes) ...");
    let status = Command::new("llama-imatrix")
        .arg("-m")
        .arg(model)
        .arg("-f")
        .arg(&calib)
        .arg("-o")
        .arg(&out)
        .args(["--chunks", "60", "-ngl", "99"])
        .status()?;
    if !status.success() {
        eprintln!("llama-imatrix failed; continuing without an imatrix");
        return Ok(None);
    }
    Ok(Some(out))
}
