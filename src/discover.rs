//! `shoehorn discover`: which models are worth fitting to this machine.
//!
//! Pulls the most-downloaded GGUF text-generation repos from Hugging Face,
//! keeps the ones that publish a full-precision source (the same
//! BF16 > F16 > F32 selection `fit` uses), and estimates the bits/weight
//! this machine's budget affords each one. The estimate is deliberately
//! rough — `shoehorn fit <repo> --dry-run` gives the exact solve.

use anyhow::{anyhow, Context, Result};
use rayon::prelude::*;
use std::process::Command;

pub struct Suggestion {
    pub repo: String,
    pub downloads: u64,
    /// total bytes of the BF16/F16/F32 source (all shards)
    pub src_bytes: u64,
    /// estimated parameter count (source bytes / bytes-per-param)
    pub params: f64,
    /// estimated achievable bits/weight inside the budget
    pub bpw: f64,
    pub verdict: &'static str,
    /// full-precision file is far smaller than the repo name claims
    pub suspicious: bool,
}

/// Overhead the weight budget loses to KV + compute + reserve, without the
/// model's metadata to compute it exactly. The KV term (~160 KiB/token of
/// context) matches a 14B-class dense model at f16 and overshoots small
/// ones, which keeps the estimate conservative.
pub fn est_overhead(ctx: u64) -> u64 {
    let reserve = 512u64 << 20;
    let compute = 320u64 << 20;
    let kv = ctx * 160 * 1024;
    reserve + compute + kv
}

pub fn verdict_for(bpw: f64) -> &'static str {
    match bpw {
        b if b >= 16.0 => "runs at full precision",
        b if b >= 8.5 => "near-lossless (Q8-class)",
        b if b >= 6.0 => "excellent (Q6-class)",
        b if b >= 4.5 => "good (Q4/Q5-class)",
        b if b >= 3.0 => "tight — worthwhile for big models",
        b if b >= 2.06 => "at the floor, heavily degraded",
        _ => "does not fit",
    }
}

/// Bytes per parameter of the source file: BF16/F16 are 2, F32 is 4.
fn bytes_per_param(files: &[String]) -> f64 {
    let f32_src = files
        .first()
        .map(|f| {
            let l = f.to_lowercase();
            l.contains("f32") && !l.contains("f16")
        })
        .unwrap_or(false);
    if f32_src { 4.0 } else { 2.0 }
}

/// Largest "<number>B" parameter claim in a repo name ("Qwen3.8-27B-GGUF"
/// → 27.0), used to spot repos whose full-precision file is actually a
/// small draft companion rather than the named model.
fn name_params_hint(repo: &str) -> Option<f64> {
    let s = repo.as_bytes();
    let mut best: Option<f64> = None;
    let mut i = 0;
    while i < s.len() {
        if s[i].is_ascii_digit() && (i == 0 || !s[i - 1].is_ascii_alphabetic()) {
            let start = i;
            while i < s.len() && (s[i].is_ascii_digit() || s[i] == b'.') {
                i += 1;
            }
            let num_end = i;
            if num_end < s.len()
                && (s[num_end] == b'b' || s[num_end] == b'B')
                && (num_end + 1 == s.len() || !s[num_end + 1].is_ascii_alphanumeric())
                && let Ok(v) = repo[start..num_end].trim_matches('.').parse::<f64>()
            {
                best = Some(best.map_or(v, |b: f64| b.max(v)));
            }
        } else {
            i += 1;
        }
    }
    best
}

fn popular_gguf_repos(limit: usize) -> Result<Vec<(String, u64)>> {
    let api = format!(
        "https://huggingface.co/api/models?filter=gguf&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={limit}"
    );
    let out = Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "15", &api])
        .output()
        .context("running curl")?;
    if !out.status.success() {
        anyhow::bail!("Hugging Face API request failed (network?)");
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    Ok(v.as_array()
        .ok_or_else(|| anyhow!("unexpected API response shape"))?
        .iter()
        .filter_map(|m| {
            Some((
                m.get("id")?.as_str()?.to_string(),
                m.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0),
            ))
        })
        .collect())
}

/// Rank fit-worthy repos for a machine with `usable_vram` at context `ctx`.
pub fn discover(usable_vram: u64, ctx: u64, scan: usize) -> Result<Vec<Suggestion>> {
    let weight_budget = usable_vram.saturating_sub(est_overhead(ctx));
    if weight_budget == 0 {
        anyhow::bail!("no room for weights at this budget/ctx");
    }
    let repos = popular_gguf_repos(scan)?;
    eprintln!("scanning {} popular GGUF repos for full-precision sources ...", repos.len());
    let mut out: Vec<Suggestion> = repos
        .par_iter()
        .filter_map(|(id, downloads)| {
            let (owner, repo) = id.split_once('/')?;
            let entries = crate::fetch::repo_entries(owner, repo).ok()?;
            let (files, total) = crate::fetch::select_model_files(&entries).ok()?;
            if total == 0 {
                return None;
            }
            let params = total as f64 / bytes_per_param(&files);
            // more budget than the source has bits is just "full precision"
            let bpw = (weight_budget as f64 * 8.0 / params).min(16.0);
            // e.g. a "27B" repo whose only F16 is 900 MB: a draft model
            let suspicious = name_params_hint(id)
                .is_some_and(|hint| params / 1e9 < hint * 0.3);
            Some(Suggestion {
                repo: id.clone(),
                downloads: *downloads,
                src_bytes: total,
                params,
                bpw,
                verdict: if suspicious {
                    "source is far smaller than the name — likely a draft companion"
                } else {
                    verdict_for(bpw)
                },
                suspicious,
            })
        })
        .collect();
    // Biggest model that still fits well makes the best suggestion: sort by
    // achievable-quality tier first, then by parameter count within a tier.
    out.retain(|s| s.bpw >= 2.06);
    out.sort_by(|a, b| {
        let tier = |s: &Suggestion| match s.bpw {
            _ if s.suspicious => -1,
            x if x >= 4.5 => 2,
            x if x >= 3.0 => 1,
            _ => 0,
        };
        tier(b).cmp(&tier(a)).then(
            b.params.partial_cmp(&a.params).unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_tiers() {
        assert_eq!(verdict_for(16.5), "runs at full precision");
        assert_eq!(verdict_for(7.3), "excellent (Q6-class)");
        assert_eq!(verdict_for(4.7), "good (Q4/Q5-class)");
        assert_eq!(verdict_for(3.4), "tight — worthwhile for big models");
        assert_eq!(verdict_for(1.5), "does not fit");
    }

    #[test]
    fn name_hints_parse() {
        assert_eq!(name_params_hint("owner/Qwen3.8-27B-GGUF"), Some(27.0));
        assert_eq!(name_params_hint("owner/LFM2.5-2.6B-GGUF"), Some(2.6));
        assert_eq!(name_params_hint("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"), Some(8.0));
        assert_eq!(name_params_hint("owner/maple-preview-GGUF"), None);
    }

    #[test]
    fn overhead_scales_with_ctx() {
        assert!(est_overhead(8192) > est_overhead(2048));
        // ~2 GiB at 8k context: 512 MiB reserve + 320 MiB compute + 1.25 GiB KV
        let at_8k = est_overhead(8192);
        assert!(at_8k > (1u64 << 31) - (200 << 20) && at_8k < (1u64 << 31) + (200 << 20));
    }
}
