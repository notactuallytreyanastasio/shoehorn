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

/// Choose which repo files make up the model: the largest BF16 GGUF, then
/// F16, then F32 — expanded to every sibling shard (sorted) when the pick is
/// a llama.cpp-style split (`-00001-of-0000N.gguf`). Returns the paths and
/// their total size.
pub fn select_model_files(entries: &[(String, u64)]) -> Result<(Vec<String>, u64)> {
    let pick = |pred: &dyn Fn(&str) -> bool| -> Option<&(String, u64)> {
        entries
            .iter()
            .filter(|(p, _)| {
                let l = p.to_lowercase();
                // mmproj files are vision projectors that ride along in
                // multimodal repos, not the model itself
                p.ends_with(".gguf")
                    && !l.rsplit('/').next().unwrap_or(&l).starts_with("mmproj")
                    && pred(&l)
            })
            .max_by_key(|(_, s)| *s)
    };
    let model_file = pick(&|p| p.contains("bf16"))
        .or_else(|| pick(&|p| p.contains("f16") && !p.contains("bf16")))
        .or_else(|| pick(&|p| p.contains("f32")))
        .ok_or_else(|| anyhow!("no BF16/F16/F32 GGUF"))?;
    let stem = &model_file.0;
    let shards: Vec<&(String, u64)> = if let Some((prefix, _)) = stem.split_once("-of-") {
        let prefix = prefix.rsplit_once('-').map(|(p, _)| p).unwrap_or(stem);
        let mut v: Vec<&(String, u64)> = entries
            .iter()
            .filter(|(p, _)| p.starts_with(prefix) && p.contains("-of-") && p.ends_with(".gguf"))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    } else {
        vec![model_file]
    };
    let total = shards.iter().map(|(_, s)| s).sum();
    Ok((shards.into_iter().map(|(p, _)| p.clone()).collect(), total))
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

/// List a repo's files (path, size) via the Hugging Face tree API.
pub fn repo_entries(owner: &str, repo: &str) -> Result<Vec<(String, u64)>> {
    let api = format!("https://huggingface.co/api/models/{owner}/{repo}/tree/main");
    let out = Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "15", &api])
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("Hugging Face API request failed for {owner}/{repo} (repo exists and is public?)");
    }
    let files: serde_json::Value = serde_json::from_slice(&out.stdout)
        .context("parsing Hugging Face API response")?;
    Ok(files
        .as_array()
        .ok_or_else(|| anyhow!("unexpected API response shape"))?
        .iter()
        .filter_map(|f| {
            Some((
                f.get("path")?.as_str()?.to_string(),
                f.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
            ))
        })
        .collect())
}

fn hf_repo(owner: &str, repo: &str) -> Result<Resolved> {
    let entries = repo_entries(owner, repo)?;

    let (shard_files, total) = select_model_files(&entries).map_err(|e| {
        anyhow!(
            "{e} in {owner}/{repo}; files: {}",
            entries
                .iter()
                .filter(|(p, _)| p.ends_with(".gguf"))
                .map(|(p, _)| p.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let dir = cache_dir()?.join(format!("{owner}__{repo}"));
    eprintln!(
        "fetching {}/{} ({} file(s), {:.1} GB) ...",
        owner,
        shard_files[0],
        shard_files.len(),
        total as f64 / 1e9
    );
    let mut model = None;
    for p in &shard_files {
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
        std::fs::create_dir_all(cache_dir()?)?;
        // Community-standard mixed corpus first: on Qwen3-0.6B it measured
        // ~1% lower neutral-text PPL than man-page calibration at the same
        // budget. Man pages remain the offline fallback; neither working
        // just means no auto-imatrix, not a failed fit.
        let url = "https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/calibration_datav3.txt";
        let fetched = Command::new("curl")
            .args(["-sL", "--fail", "--max-time", "30", "-o"])
            .arg(&calib)
            .arg(url)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !fetched || calib.metadata().map(|m| m.len() < 100_000).unwrap_or(true) {
            let _ = std::fs::remove_file(&calib);
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

#[cfg(test)]
mod tests {
    use super::select_model_files;

    fn e(items: &[(&str, u64)]) -> Vec<(String, u64)> {
        items.iter().map(|(p, s)| (p.to_string(), *s)).collect()
    }

    #[test]
    fn prefers_bf16_then_f16_then_f32() {
        let entries = e(&[
            ("m-Q4_K_M.gguf", 500),
            ("m-F32.gguf", 4000),
            ("m-F16.gguf", 2000),
            ("m-BF16.gguf", 1999),
        ]);
        let (files, total) = select_model_files(&entries).unwrap();
        assert_eq!(files, vec!["m-BF16.gguf"]);
        assert_eq!(total, 1999);

        let entries = e(&[("m-F32.gguf", 4000), ("m-F16.gguf", 2000)]);
        assert_eq!(select_model_files(&entries).unwrap().0, vec!["m-F16.gguf"]);

        let entries = e(&[("m-F32.gguf", 4000), ("m-Q8_0.gguf", 1000)]);
        assert_eq!(select_model_files(&entries).unwrap().0, vec!["m-F32.gguf"]);
    }

    #[test]
    fn expands_split_models_sorted() {
        let entries = e(&[
            ("BF16/m-BF16-00002-of-00003.gguf", 10),
            ("m-Q4_K_M.gguf", 500),
            ("BF16/m-BF16-00003-of-00003.gguf", 5),
            ("BF16/m-BF16-00001-of-00003.gguf", 10),
        ]);
        let (files, total) = select_model_files(&entries).unwrap();
        assert_eq!(
            files,
            vec![
                "BF16/m-BF16-00001-of-00003.gguf",
                "BF16/m-BF16-00002-of-00003.gguf",
                "BF16/m-BF16-00003-of-00003.gguf",
            ]
        );
        assert_eq!(total, 25);
    }

    #[test]
    fn errors_without_full_precision_candidate() {
        let entries = e(&[("m-Q4_K_M.gguf", 500), ("readme.md", 1)]);
        assert!(select_model_files(&entries).is_err());
    }

    #[test]
    fn ignores_mmproj_projector_files() {
        let entries = e(&[("mmproj-model-F16.gguf", 900), ("m-BF16.gguf", 500)]);
        assert_eq!(select_model_files(&entries).unwrap().0, vec!["m-BF16.gguf"]);
        // a repo with ONLY an mmproj has no usable source
        let entries = e(&[("sub/mmproj-F16.gguf", 900), ("m-Q4_K_M.gguf", 500)]);
        assert!(select_model_files(&entries).is_err());
    }
}
