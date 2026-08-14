//! Imatrix loading: legacy binary format and the newer GGUF-based format.
//!
//! Both yield, per tensor name, a vector of per-column importance weights
//! (mean squared activation per column).

use crate::gguf;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;

pub type Imatrix = HashMap<String, Vec<f32>>;

pub fn load(path: &str) -> Result<Imatrix> {
    let buf = std::fs::read(path).with_context(|| format!("reading imatrix {path}"))?;
    if buf.len() >= 4 && buf[0..4] == *b"GGUF" {
        load_gguf(&buf)
    } else {
        load_legacy(&buf)
    }
}

/// New format: GGUF file with `<tensor>.in_sum2` (f32 sums of squared
/// activations) and `<tensor>.counts` (f32 chunk counts) tensors.
fn load_gguf(buf: &[u8]) -> Result<Imatrix> {
    let f = gguf::read(buf)?;
    let mut sums: HashMap<String, Vec<f32>> = HashMap::new();
    let mut counts: HashMap<String, Vec<f32>> = HashMap::new();
    for t in &f.tensors {
        let data = f.tensor_data(buf, t);
        let vals: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if let Some(name) = t.name.strip_suffix(".in_sum2") {
            sums.insert(name.to_string(), vals);
        } else if let Some(name) = t.name.strip_suffix(".counts") {
            counts.insert(name.to_string(), vals);
        }
    }
    if sums.is_empty() {
        bail!("GGUF imatrix contains no *.in_sum2 tensors");
    }
    let mut out = Imatrix::new();
    for (name, mut s) in sums {
        if let Some(c) = counts.get(&name)
            && !c.is_empty()
        {
            let ratio = s.len() / c.len().max(1);
            for (i, v) in s.iter_mut().enumerate() {
                let cnt = c[(i / ratio.max(1)).min(c.len() - 1)];
                if cnt > 0.0 {
                    *v /= cnt;
                }
            }
        }
        sanitize(&mut s);
        out.insert(name, s);
    }
    Ok(out)
}

/// Legacy format: i32 n_entries, then per entry
/// { i32 name_len, name, i32 ncall, i32 nval, f32 values[nval] }.
fn load_legacy(buf: &[u8]) -> Result<Imatrix> {
    let mut pos = 0usize;
    let rd_i32 = |pos: &mut usize| -> Result<i32> {
        if *pos + 4 > buf.len() {
            bail!("legacy imatrix truncated at {pos:?}");
        }
        let v = i32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let n_entries = rd_i32(&mut pos)?;
    if !(0..1_000_000).contains(&n_entries) {
        bail!("implausible legacy imatrix entry count {n_entries}; not an imatrix file?");
    }
    let mut out = Imatrix::new();
    for _ in 0..n_entries {
        let name_len = rd_i32(&mut pos)? as usize;
        if pos + name_len > buf.len() {
            bail!("legacy imatrix truncated in name");
        }
        let name = String::from_utf8_lossy(&buf[pos..pos + name_len]).into_owned();
        pos += name_len;
        let ncall = rd_i32(&mut pos)?;
        let nval = rd_i32(&mut pos)? as usize;
        if pos + nval * 4 > buf.len() {
            bail!("legacy imatrix truncated in values for {name}");
        }
        let mut vals: Vec<f32> = buf[pos..pos + nval * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        pos += nval * 4;
        if ncall > 0 {
            for v in &mut vals {
                *v /= ncall as f32;
            }
        }
        sanitize(&mut vals);
        out.insert(name, vals);
    }
    Ok(out)
}

/// Guard against zeros/negatives/NaNs that would null out the weighted fit.
fn sanitize(vals: &mut [f32]) {
    let mut maxv = 0f32;
    for v in vals.iter() {
        if v.is_finite() && *v > maxv {
            maxv = *v;
        }
    }
    let floor = if maxv > 0.0 { maxv * 1e-9 } else { 1.0 };
    for v in vals.iter_mut() {
        if !v.is_finite() || *v <= 0.0 {
            *v = floor;
        }
    }
}

/// Weight slice for row `row_idx` of a tensor with row length `ne0` and
/// `n_mats` matrices (ne2 for 3D experts): imatrix may cover ne0 or ne0*n_mats.
pub fn row_slice(
    im: Option<&[f32]>,
    ne0: usize,
    rows_per_mat: usize,
    row_idx: usize,
) -> Option<&[f32]> {
    let im = im?;
    if im.len() == ne0 {
        Some(im)
    } else if rows_per_mat > 0 && im.len() % ne0 == 0 {
        let mat = row_idx / rows_per_mat;
        let n_mats = im.len() / ne0;
        let m = mat.min(n_mats - 1);
        Some(&im[m * ne0..(m + 1) * ne0])
    } else {
        None
    }
}
