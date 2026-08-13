//! Importance-weighted quantization kernels producing llama.cpp-compatible blocks.
//!
//! The scale-search logic mirrors ggml's `make_qx_quants` / `make_qkx3_quants`
//! (weighted least squares over a family of candidate grids), with the element
//! weight w[j] = imatrix[j] * sqrt(sigma2 + x[j]^2) as in ggml's
//! `quantize_row_*_impl` functions.

use crate::gguf::GgmlType;
use half::f16;

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

#[inline]
fn f16_bytes(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_le_bytes()
}

#[inline]
fn f16_val(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

const GROUP_MAX_EPS: f32 = 1e-15;

/// Weighted symmetric grid search: find scale d so x ~ d * q, q in [-nmax, nmax-1].
/// Writes q + nmax into `ls`. Returns d.
fn make_qx_quants(x: &[f32], nmax: i32, w: &[f32], ls: &mut [u8]) -> f32 {
    let n = x.len();
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        ls[..n].fill(nmax as u8);
        return 0.0;
    }
    let mut best_scale = 0f32;
    let mut best = 0f32;
    let mut first = true;
    for is in -9i32..=9 {
        let iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let mut sumlx = 0f32;
        let mut suml2 = 0f32;
        let mut cand = vec![0u8; n];
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            cand[i] = (l + nmax) as u8;
            let wi = w[i];
            sumlx += wi * x[i] * l as f32;
            suml2 += wi * (l * l) as f32;
        }
        if suml2 > 0.0 && (first || sumlx * sumlx > best * suml2) {
            let scale = sumlx / suml2;
            best_scale = scale;
            best = scale * sumlx;
            ls[..n].copy_from_slice(&cand);
            first = false;
        }
    }
    best_scale
}

/// Weighted asymmetric search: x ~ d * q + min, q in [0, nmax].
/// Returns (d, min); min is <= 0. Mirrors ggml make_qkx3_quants.
fn make_qkx3_quants(
    x: &[f32],
    nmax: i32,
    w: &[f32],
    ls: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut vmin = x.iter().cloned().fold(f32::INFINITY, f32::min).min(0.0);
    let vmax = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if vmax <= vmin {
        ls[..n].fill(0);
        return (0.0, vmin);
    }
    let mut iscale = nmax as f32 / (vmax - vmin);
    let mut scale = 1.0 / iscale;
    let mut best_err = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - vmin)).clamp(0, nmax);
        ls[i] = l as u8;
        let diff = scale * l as f32 + vmin - x[i];
        best_err += w[i] * diff * diff;
    }
    let mut best_min = vmin;
    let mut laux = vec![0u8; n];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (vmax - vmin);
        let (mut sum_w, mut sum_x, mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32, 0f32, 0f32);
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - vmin)).clamp(0, nmax);
            laux[i] = l as u8;
            let wi = w[i];
            sum_w += wi;
            sum_x += wi * x[i];
            sum_l += wi * l as f32;
            sum_l2 += wi * (l * l) as f32;
            sum_xl += wi * l as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = if sum_l2 > 0.0 { sum_xl / sum_l2 } else { 0.0 };
            }
            let mut err = 0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                err += w[i] * diff * diff;
            }
            if err < best_err {
                best_err = err;
                scale = this_scale;
                best_min = this_min;
                ls[..n].copy_from_slice(&laux);
            }
        }
    }
    // vmin may have been replaced by a better fitted min
    vmin = best_min;
    (scale, vmin)
}

/// Element weights: imatrix weight (or 1.0) shaped by sqrt(sigma2 + x^2).
fn element_weights(x: &[f32], im: Option<&[f32]>, sigma2: f32) -> Vec<f32> {
    let mut w = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        let base = im.map_or(1.0, |m| m[i]);
        w.push(base * (sigma2 + v * v).sqrt());
    }
    w
}

fn row_sigma2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32
}

// ---------------- per-row encoders ----------------

fn enc_q8_0(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.chunks(32) {
        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16_bytes(d));
        for &v in blk {
            out.push(nearest_int(v * id).clamp(-127, 127) as i8 as u8);
        }
    }
}

fn enc_q4_0(x: &[f32], im: Option<&[f32]>, sigma2: f32, out: &mut Vec<u8>) {
    let mut ls = [0u8; 32];
    for (bi, blk) in x.chunks(32).enumerate() {
        let w = element_weights(blk, im.map(|m| &m[bi * 32..bi * 32 + 32]), sigma2);
        let d = make_qx_quants(blk, 8, &w, &mut ls);
        out.extend_from_slice(&f16_bytes(d));
        for j in 0..16 {
            out.push((ls[j] & 0xF) | (ls[j + 16] << 4));
        }
    }
}

fn enc_q5_0(x: &[f32], im: Option<&[f32]>, sigma2: f32, out: &mut Vec<u8>) {
    let mut ls = [0u8; 32];
    for (bi, blk) in x.chunks(32).enumerate() {
        let w = element_weights(blk, im.map(|m| &m[bi * 32..bi * 32 + 32]), sigma2);
        let d = make_qx_quants(blk, 16, &w, &mut ls);
        out.extend_from_slice(&f16_bytes(d));
        let mut qh = 0u32;
        for j in 0..32 {
            qh |= (((ls[j] >> 4) & 1) as u32) << j;
        }
        out.extend_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            out.push((ls[j] & 0xF) | ((ls[j + 16] & 0xF) << 4));
        }
    }
}

fn enc_q4_1(x: &[f32], im: Option<&[f32]>, sigma2: f32, out: &mut Vec<u8>) {
    let mut ls = [0u8; 32];
    for (bi, blk) in x.chunks(32).enumerate() {
        let w = element_weights(blk, im.map(|m| &m[bi * 32..bi * 32 + 32]), sigma2);
        let (d, min) = make_qkx3_quants(blk, 15, &w, &mut ls, -0.9, 0.05, 36);
        out.extend_from_slice(&f16_bytes(d));
        out.extend_from_slice(&f16_bytes(min));
        for j in 0..16 {
            out.push((ls[j] & 0xF) | (ls[j + 16] << 4));
        }
    }
}

fn enc_q5_1(x: &[f32], im: Option<&[f32]>, sigma2: f32, out: &mut Vec<u8>) {
    let mut ls = [0u8; 32];
    for (bi, blk) in x.chunks(32).enumerate() {
        let w = element_weights(blk, im.map(|m| &m[bi * 32..bi * 32 + 32]), sigma2);
        let (d, min) = make_qkx3_quants(blk, 31, &w, &mut ls, -0.9, 0.05, 36);
        out.extend_from_slice(&f16_bytes(d));
        out.extend_from_slice(&f16_bytes(min));
        let mut qh = 0u32;
        for j in 0..32 {
            qh |= (((ls[j] >> 4) & 1) as u32) << j;
        }
        out.extend_from_slice(&qh.to_le_bytes());
        for j in 0..16 {
            out.push((ls[j] & 0xF) | ((ls[j + 16] & 0xF) << 4));
        }
    }
}

fn enc_q6_k(x: &[f32], im: Option<&[f32]>, sigma2: f32, out: &mut Vec<u8>) {
    let mut l_all = [0u8; 256];
    for (bi, blk) in x.chunks(256).enumerate() {
        let mut scales = [0f32; 16];
        let mut max_abs_scale = 0f32;
        let mut max_scale = 0f32;
        for sb in 0..16 {
            let xs = &blk[sb * 16..sb * 16 + 16];
            let w = element_weights(xs, im.map(|m| &m[bi * 256 + sb * 16..bi * 256 + sb * 16 + 16]), sigma2);
            let mut ls = [0u8; 16];
            let s = make_qx_quants(xs, 32, &w, &mut ls);
            scales[sb] = s;
            if s.abs() > max_abs_scale {
                max_abs_scale = s.abs();
                max_scale = s;
            }
        }
        let base = out.len();
        out.resize(base + 210, 0);
        if max_abs_scale < GROUP_MAX_EPS {
            continue; // all-zero block
        }
        let iscale = -128.0 / max_scale;
        let d = 1.0 / iscale;
        let mut qscales = [0i8; 16];
        for sb in 0..16 {
            qscales[sb] = nearest_int(iscale * scales[sb]).min(127) as i8;
        }
        for sb in 0..16 {
            let dd = d * qscales[sb] as f32;
            if dd == 0.0 {
                for j in 0..16 {
                    l_all[sb * 16 + j] = 32;
                }
                continue;
            }
            for j in 0..16 {
                let l = nearest_int(blk[sb * 16 + j] / dd).clamp(-32, 31);
                l_all[sb * 16 + j] = (l + 32) as u8;
            }
        }
        // pack: two halves of 128
        for half_i in 0..2 {
            let lo = &l_all[half_i * 128..half_i * 128 + 128];
            let qlo = base + half_i * 64;
            let qho = base + 128 + half_i * 32;
            for l in 0..32 {
                out[qlo + l] = (lo[l] & 0xF) | ((lo[l + 64] & 0xF) << 4);
                out[qlo + 32 + l] = (lo[l + 32] & 0xF) | ((lo[l + 96] & 0xF) << 4);
                out[qho + l] = (lo[l] >> 4)
                    | ((lo[l + 32] >> 4) << 2)
                    | ((lo[l + 64] >> 4) << 4)
                    | ((lo[l + 96] >> 4) << 6);
            }
        }
        for sb in 0..16 {
            out[base + 192 + sb] = qscales[sb] as u8;
        }
        out[base + 208..base + 210].copy_from_slice(&f16_bytes(d));
    }
}

/// Shared Q4_K / Q5_K encoder. nmax = 15 or 31.
fn enc_qk45(x: &[f32], im: Option<&[f32]>, sigma2: f32, nmax: i32, out: &mut Vec<u8>) {
    let five = nmax == 31;
    let block_bytes = if five { 176 } else { 144 };
    let mut l_all = [0u8; 256];
    for (bi, blk) in x.chunks(256).enumerate() {
        let mut scales = [0f32; 8];
        let mut mins = [0f32; 8];
        for sb in 0..8 {
            let xs = &blk[sb * 32..sb * 32 + 32];
            let w = element_weights(xs, im.map(|m| &m[bi * 256 + sb * 32..bi * 256 + sb * 32 + 32]), sigma2);
            let mut ls = [0u8; 32];
            let (s, mn) = make_qkx3_quants(xs, nmax, &w, &mut ls, -0.9, 0.05, 36);
            scales[sb] = s;
            mins[sb] = -mn; // store positive; dequant is d*q - dmin*m
        }
        let max_scale = scales.iter().fold(0f32, |a, &v| a.max(v));
        let max_min = mins.iter().fold(0f32, |a, &v| a.max(v));
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut sc_bytes = [0u8; 12];
        let mut lsq = [0u8; 8];
        let mut lmq = [0u8; 8];
        for sb in 0..8 {
            lsq[sb] = (nearest_int(inv_scale * scales[sb]).clamp(0, 63)) as u8;
            lmq[sb] = (nearest_int(inv_min * mins[sb]).clamp(0, 63)) as u8;
            if sb < 4 {
                sc_bytes[sb] = lsq[sb];
                sc_bytes[sb + 4] = lmq[sb];
            } else {
                sc_bytes[sb + 4] = (lsq[sb] & 0xF) | ((lmq[sb] & 0xF) << 4);
                sc_bytes[sb - 4] |= (lsq[sb] >> 4) << 6;
                sc_bytes[sb] |= (lmq[sb] >> 4) << 6;
            }
        }
        let d = max_scale / 63.0;
        let dmin = max_min / 63.0;
        // requantize with the quantized scales
        for sb in 0..8 {
            let dsub = d * lsq[sb] as f32;
            let msub = dmin * lmq[sb] as f32;
            if dsub == 0.0 {
                for j in 0..32 {
                    l_all[sb * 32 + j] = 0;
                }
                continue;
            }
            for j in 0..32 {
                let l = nearest_int((blk[sb * 32 + j] + msub) / dsub).clamp(0, nmax);
                l_all[sb * 32 + j] = l as u8;
            }
        }
        let base = out.len();
        out.resize(base + block_bytes, 0);
        out[base..base + 2].copy_from_slice(&f16_bytes(d));
        out[base + 2..base + 4].copy_from_slice(&f16_bytes(dmin));
        out[base + 4..base + 16].copy_from_slice(&sc_bytes);
        if five {
            // qh then qs
            let qh = base + 16;
            let qs = base + 48;
            for chunk in 0..4 {
                let u1 = 1u8 << (chunk * 2);
                let u2 = 2u8 << (chunk * 2);
                for l in 0..32 {
                    let a = l_all[chunk * 64 + l];
                    let b = l_all[chunk * 64 + 32 + l];
                    out[qs + chunk * 32 + l] = (a & 0xF) | ((b & 0xF) << 4);
                    if a >= 16 {
                        out[qh + l] |= u1;
                    }
                    if b >= 16 {
                        out[qh + l] |= u2;
                    }
                }
            }
        } else {
            let qs = base + 16;
            for chunk in 0..4 {
                for l in 0..32 {
                    out[qs + chunk * 32 + l] =
                        (l_all[chunk * 64 + l] & 0xF) | (l_all[chunk * 64 + 32 + l] << 4);
                }
            }
        }
    }
}

// ---------------- per-row decoders (for error measurement & tests) ----------------

fn dec_q8_0(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(34) {
        let d = f16_val(&blk[0..2]);
        for j in 0..32 {
            out.push(d * (blk[2 + j] as i8) as f32);
        }
    }
}

fn dec_q4_0(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(18) {
        let d = f16_val(&blk[0..2]);
        let qs = &blk[2..18];
        for half_i in 0..2 {
            for j in 0..16 {
                let q = if half_i == 0 { qs[j] & 0xF } else { qs[j] >> 4 };
                out.push(d * (q as i32 - 8) as f32);
            }
        }
    }
}

fn dec_q5_0(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(22) {
        let d = f16_val(&blk[0..2]);
        let qh = u32::from_le_bytes(blk[2..6].try_into().unwrap());
        let qs = &blk[6..22];
        for half_i in 0..2 {
            for j in 0..16 {
                let idx = half_i * 16 + j;
                let lo = if half_i == 0 { qs[j] & 0xF } else { qs[j] >> 4 };
                let q = lo as u32 | (((qh >> idx) & 1) << 4);
                out.push(d * (q as i32 - 16) as f32);
            }
        }
    }
}

fn dec_q4_1(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(20) {
        let d = f16_val(&blk[0..2]);
        let m = f16_val(&blk[2..4]);
        let qs = &blk[4..20];
        for half_i in 0..2 {
            for j in 0..16 {
                let q = if half_i == 0 { qs[j] & 0xF } else { qs[j] >> 4 };
                out.push(d * q as f32 + m);
            }
        }
    }
}

fn dec_q5_1(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(24) {
        let d = f16_val(&blk[0..2]);
        let m = f16_val(&blk[2..4]);
        let qh = u32::from_le_bytes(blk[4..8].try_into().unwrap());
        let qs = &blk[8..24];
        for half_i in 0..2 {
            for j in 0..16 {
                let idx = half_i * 16 + j;
                let lo = if half_i == 0 { qs[j] & 0xF } else { qs[j] >> 4 };
                let q = lo as u32 | (((qh >> idx) & 1) << 4);
                out.push(d * q as f32 + m);
            }
        }
    }
}

fn dec_q6_k(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(210) {
        let d = f16_val(&blk[208..210]);
        let mut y = [0f32; 256];
        for half_i in 0..2 {
            let ql = &blk[half_i * 64..half_i * 64 + 64];
            let qh = &blk[128 + half_i * 32..128 + half_i * 32 + 32];
            let sc = &blk[192 + half_i * 8..192 + half_i * 8 + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] as i32) & 3) << 4)) - 32;
                let q2 = ((ql[l + 32] & 0xF) as i32 | ((((qh[l] as i32) >> 2) & 3) << 4)) - 32;
                let q3 = ((ql[l] >> 4) as i32 | ((((qh[l] as i32) >> 4) & 3) << 4)) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | ((((qh[l] as i32) >> 6) & 3) << 4)) - 32;
                let o = half_i * 128;
                y[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
        out.extend_from_slice(&y);
    }
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn dec_q4_k(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(144) {
        let d = f16_val(&blk[0..2]);
        let dmin = f16_val(&blk[2..4]);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(chunk * 2, scales);
            let (sc2, m2) = get_scale_min_k4(chunk * 2 + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            for l in 0..32 {
                out.push(d1 * (qs[chunk * 32 + l] & 0xF) as f32 - mm1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[chunk * 32 + l] >> 4) as f32 - mm2);
            }
        }
    }
}

fn dec_q5_k(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(176) {
        let d = f16_val(&blk[0..2]);
        let dmin = f16_val(&blk[2..4]);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(chunk * 2, scales);
            let (sc2, m2) = get_scale_min_k4(chunk * 2 + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let u1 = 1u8 << (chunk * 2);
            let u2 = 2u8 << (chunk * 2);
            for l in 0..32 {
                let q = (qs[chunk * 32 + l] & 0xF) as f32 + if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out.push(d1 * q - mm1);
            }
            for l in 0..32 {
                let q = (qs[chunk * 32 + l] >> 4) as f32 + if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out.push(d2 * q - mm2);
            }
        }
    }
}

// ---------------- public API ----------------

/// Quantize one row of f32 values into `ty`, appending to `out`.
/// `im` is the per-column importance slice (len == row len) if available.
pub fn encode_row(ty: GgmlType, x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    let sigma2 = row_sigma2(x);
    match ty {
        GgmlType::F32 => {
            for &v in x {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        GgmlType::F16 => {
            for &v in x {
                out.extend_from_slice(&f16_bytes(v));
            }
        }
        GgmlType::Bf16 => {
            for &v in x {
                out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        GgmlType::Q8_0 => enc_q8_0(x, out),
        GgmlType::Q4_0 => enc_q4_0(x, im, sigma2, out),
        GgmlType::Q4_1 => enc_q4_1(x, im, sigma2, out),
        GgmlType::Q5_0 => enc_q5_0(x, im, sigma2, out),
        GgmlType::Q5_1 => enc_q5_1(x, im, sigma2, out),
        GgmlType::Q6K => enc_q6_k(x, im, sigma2, out),
        GgmlType::Q4K => enc_qk45(x, im, sigma2, 15, out),
        GgmlType::Q5K => enc_qk45(x, im, sigma2, 31, out),
        GgmlType::Iq2Xxs => crate::quant_iq::enc_iq2_xxs(x, im, out),
        GgmlType::Iq2Xs => crate::quant_iq::enc_iq2_xs(x, im, out),
        GgmlType::Iq2S => crate::quant_iq::enc_iq2_s(x, im, out),
        GgmlType::Iq3Xxs => crate::quant_iq::enc_iq3_xxs(x, im, out),
        GgmlType::Iq3S => crate::quant_iq::enc_iq3_s(x, im, out),
        GgmlType::Iq4Nl => crate::quant_iq::enc_iq4_nl(x, im, out),
        GgmlType::Iq4Xs => crate::quant_iq::enc_iq4_xs(x, im, out),
        GgmlType::Other(v) => panic!("cannot encode ggml type {v}"),
    }
}

/// Decode one row previously encoded with `ty`.
pub fn decode_row(ty: GgmlType, data: &[u8], n: usize, out: &mut Vec<f32>) {
    match ty {
        GgmlType::F32 => {
            for c in data.chunks(4).take(n) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        GgmlType::F16 => {
            for c in data.chunks(2).take(n) {
                out.push(f16_val(c));
            }
        }
        GgmlType::Bf16 => {
            for c in data.chunks(2).take(n) {
                out.push(bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        GgmlType::Q8_0 => dec_q8_0(data, out),
        GgmlType::Q4_0 => dec_q4_0(data, out),
        GgmlType::Q4_1 => dec_q4_1(data, out),
        GgmlType::Q5_0 => dec_q5_0(data, out),
        GgmlType::Q5_1 => dec_q5_1(data, out),
        GgmlType::Q6K => dec_q6_k(data, out),
        GgmlType::Q4K => dec_q4_k(data, out),
        GgmlType::Q5K => dec_q5_k(data, out),
        GgmlType::Iq2Xxs => crate::quant_iq::dec_iq2_xxs(data, out),
        GgmlType::Iq2Xs => crate::quant_iq::dec_iq2_xs(data, out),
        GgmlType::Iq2S => crate::quant_iq::dec_iq2_s(data, out),
        GgmlType::Iq3Xxs => crate::quant_iq::dec_iq3_xxs(data, out),
        GgmlType::Iq3S => crate::quant_iq::dec_iq3_s(data, out),
        GgmlType::Iq4Nl => crate::quant_iq::dec_iq4_nl(data, out),
        GgmlType::Iq4Xs => crate::quant_iq::dec_iq4_xs(data, out),
        GgmlType::Other(v) => panic!("cannot decode ggml type {v}"),
    }
}

/// Weighted squared error of quantizing `x` to `ty` (encode+decode round trip).
pub fn row_error(ty: GgmlType, x: &[f32], im: Option<&[f32]>) -> f64 {
    let mut enc = Vec::with_capacity(ty.row_bytes(x.len() as u64) as usize);
    encode_row(ty, x, im, &mut enc);
    let mut dec = Vec::with_capacity(x.len());
    decode_row(ty, &enc, x.len(), &mut dec);
    let mut err = 0f64;
    for i in 0..x.len() {
        let w = im.map_or(1.0, |m| m[i]) as f64;
        let d = (x[i] - dec[i]) as f64;
        err += w * d * d;
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vec(n: usize) -> Vec<f32> {
        // deterministic pseudo-random values
        let mut state = 0x12345678u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32 - 0.5) * 4.0
            })
            .collect()
    }

    fn check_roundtrip(ty: GgmlType, tol: f32) {
        let x = test_vec(512);
        let mut enc = Vec::new();
        encode_row(ty, &x, None, &mut enc);
        assert_eq!(enc.len() as u64, ty.row_bytes(512));
        let mut dec = Vec::new();
        decode_row(ty, &enc, 512, &mut dec);
        assert_eq!(dec.len(), 512);
        let rmse = (x.iter().zip(&dec).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 512.0).sqrt();
        assert!(rmse < tol, "{} rmse {} exceeds {}", ty.name(), rmse, tol);
    }

    #[test]
    fn roundtrips() {
        check_roundtrip(GgmlType::F16, 1e-3);
        check_roundtrip(GgmlType::Bf16, 1e-2);
        check_roundtrip(GgmlType::Q8_0, 0.02);
        check_roundtrip(GgmlType::Q6K, 0.05);
        check_roundtrip(GgmlType::Q5K, 0.08);
        check_roundtrip(GgmlType::Q5_0, 0.1);
        check_roundtrip(GgmlType::Q5_1, 0.1);
        check_roundtrip(GgmlType::Q4K, 0.15);
        check_roundtrip(GgmlType::Q4_0, 0.25);
        check_roundtrip(GgmlType::Q4_1, 0.2);
    }

    #[test]
    fn imatrix_weighting_prioritizes_important_columns() {
        let x = test_vec(256);
        let mut im = vec![1.0f32; 256];
        for w in im.iter_mut().take(32) {
            *w = 100.0;
        }
        let err_weighted = row_error(GgmlType::Q4K, &x, Some(&im));
        let err_uniform = row_error(GgmlType::Q4K, &x, None);
        // weighted-objective error under the weighted metric should not exceed
        // the uniform encoding evaluated under the same metric
        let mut enc = Vec::new();
        encode_row(GgmlType::Q4K, &x, None, &mut enc);
        let mut dec = Vec::new();
        decode_row(GgmlType::Q4K, &enc, 256, &mut dec);
        let mut err_uniform_enc_weighted_metric = 0f64;
        for i in 0..256 {
            let d = (x[i] - dec[i]) as f64;
            err_uniform_enc_weighted_metric += im[i] as f64 * d * d;
        }
        assert!(err_weighted <= err_uniform_enc_weighted_metric * 1.05);
        assert!(err_uniform > 0.0);
    }
}
