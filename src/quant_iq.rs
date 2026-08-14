//! IQ-family codebook quantizers (IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS),
//! ported from ggml-quants.c at llama.cpp commit 48d22e295.
//!
//! IQ2/IQ3 snap groups of 8 (resp. 4) weights to points of an E8- (resp. D4-)
//! lattice-derived codebook, with signs packed under an even-parity constraint
//! (XXS/XS) or stored verbatim (S). The kmap/neighbour structures are rebuilt
//! at first use from the same grid tables llama.cpp dequantizes with.

use crate::iq_tables::*;
use half::f16;
use std::sync::OnceLock;

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
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;

// ---------------- grid/kmap/neighbour construction ----------------

pub struct GridData {
    /// each entry: `dim` codebook values (odd, 1..=7 or 1..=15 range as bytes)
    pub grid: Vec<[i8; 8]>,
    pub dim: usize,
    /// >=0: grid index; <0: -(offset+1) into neighbours
    pub map: Vec<i32>,
    /// per off-grid point: [count, idx...]
    pub neighbours: Vec<u16>,
}

fn build_grid(points: &[[i8; 8]], dim: usize, kmap_size: usize, bits: usize, nwant: usize) -> GridData {
    let grid_size = points.len();
    let mut map = vec![-1i32; kmap_size];
    for (i, p) in points.iter().enumerate() {
        let mut index = 0usize;
        for k in 0..dim {
            let q = ((p[k] - 1) / 2) as usize;
            index |= q << (bits * k);
        }
        map[index] = i as i32;
    }
    let mut neighbours = Vec::new();
    let mut dist2: Vec<(i32, u16)> = Vec::with_capacity(grid_size);
    for i in 0..kmap_size {
        if map[i] >= 0 {
            continue;
        }
        let mut pos = [0i32; 8];
        for k in 0..dim {
            let l = ((i >> (bits * k)) & ((1 << bits) - 1)) as i32;
            pos[k] = 2 * l + 1;
        }
        dist2.clear();
        for (j, p) in points.iter().enumerate() {
            let mut d2 = 0i32;
            for k in 0..dim {
                let d = p[k] as i32 - pos[k];
                d2 += d * d;
            }
            dist2.push((d2, j as u16));
        }
        dist2.sort_unstable();
        map[i] = -((neighbours.len() as i32) + 1);
        let start = neighbours.len();
        neighbours.push(0);
        let mut n = 0u16;
        let mut nhave = 1;
        let mut d2 = dist2[0].0;
        for &(d, j) in &dist2 {
            if d > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = d;
                nhave += 1;
            }
            neighbours.push(j);
            n += 1;
        }
        neighbours[start] = n;
    }
    GridData { grid: points.to_vec(), dim, map, neighbours }
}

/// Expand a packed lattice-index table (2 or 3 bits per coordinate) into
/// codebook points with odd coordinates 2l+1, exactly as iq2xs/iq3xs_init_impl.
fn expand_kgrid(packed: &[u16], dim: usize, bits: usize) -> Vec<[i8; 8]> {
    packed
        .iter()
        .map(|&v| {
            let mut p = [0i8; 8];
            for k in 0..dim {
                let l = ((v as usize) >> (bits * k)) & ((1 << bits) - 1);
                p[k] = (2 * l + 1) as i8;
            }
            p
        })
        .collect()
}

pub fn grid_iq2xxs() -> &'static GridData {
    static G: OnceLock<GridData> = OnceLock::new();
    G.get_or_init(|| build_grid(&expand_kgrid(&KGRID_2BIT_256, 8, 2), 8, 43692, 2, 2))
}
pub fn grid_iq2xs() -> &'static GridData {
    static G: OnceLock<GridData> = OnceLock::new();
    G.get_or_init(|| build_grid(&expand_kgrid(&KGRID_2BIT_512, 8, 2), 8, 43692, 2, 2))
}
pub fn grid_iq2s() -> &'static GridData {
    static G: OnceLock<GridData> = OnceLock::new();
    G.get_or_init(|| build_grid(&expand_kgrid(&KGRID_2BIT_1024, 8, 2), 8, 43692, 2, 1))
}
pub fn grid_iq3xxs() -> &'static GridData {
    static G: OnceLock<GridData> = OnceLock::new();
    G.get_or_init(|| build_grid(&expand_kgrid(&KGRID_256, 4, 3), 4, 4096, 3, 2))
}
pub fn grid_iq3s() -> &'static GridData {
    static G: OnceLock<GridData> = OnceLock::new();
    G.get_or_init(|| build_grid(&expand_kgrid(&KGRID_512, 4, 3), 4, 4096, 3, 3))
}

/// Weighted nearest-codebook-point search among the neighbour list of an
/// off-grid rounding. Writes (grid value - 1)/2 into ls. Returns grid index.
fn find_best_neighbour(
    g: &GridData,
    map_val: i32,
    xval: &[f32],
    waux: &[f32],
    scale: f32,
    ls: &mut [i8],
) -> usize {
    let off = (-map_val - 1) as usize;
    let n = g.neighbours[off] as usize;
    debug_assert!(n > 0);
    let mut best_d2 = f32::MAX;
    let mut best = usize::MAX;
    for &j in &g.neighbours[off + 1..off + 1 + n] {
        let pg = &g.grid[j as usize];
        let mut d2 = 0f32;
        for i in 0..g.dim {
            let diff = scale * pg[i] as f32 - xval[i];
            d2 += waux[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = j as usize;
        }
    }
    let pg = &g.grid[best];
    for i in 0..g.dim {
        ls[i] = (pg[i] - 1) / 2;
    }
    best
}

/// make_qp_quants: non-negative weighted quantization used to seed IQ2_XXS.
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], ls: &mut [u8], w: &[f32]) -> f32 {
    let max = x.iter().cloned().fold(0f32, f32::max);
    if max < GROUP_MAX_EPS {
        ls[..n].fill(0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    for i in 0..n {
        ls[i] = nearest_int(iscale * x[i]) as u8;
    }
    let scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let diff = x[i] - scale * ls[i] as f32;
        best_mse += w[i] * diff * diff;
    }
    for is in -4i32..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * l as f32;
            mse += w[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).min(nmax);
        ls[i] = l as u8;
        sumlx += w[i] * x[i] * l as f32;
        suml2 += w[i] * (l * l) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let wi = w[i];
            let li = ls[i] as f32;
            let slx = sumlx - wi * x[i] * li;
            let sl2 = suml2 - wi * li * li;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l as u8 != ls[i] {
                    let slx2 = slx + wi * x[i] * new_l as f32;
                    let sl22 = sl2 + wi * (new_l * new_l) as f32;
                    if slx2 * slx2 * suml2 > sumlx * sumlx * sl22 {
                        ls[i] = new_l as u8;
                        sumlx = slx2;
                        suml2 = sl22;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Split into |x| with sign byte; flip the least-important element if sign
/// parity is odd (XXS/XS store only 7 sign bits per group of 8).
fn signs_with_parity(xb: &[f32], weight: &[f32], k: usize, xval: &mut [f32]) -> u8 {
    let mut nflip = 0;
    let mut s = 0u8;
    for i in 0..8 {
        if xb[8 * k + i] >= 0.0 {
            xval[8 * k + i] = xb[8 * k + i];
        } else {
            xval[8 * k + i] = -xb[8 * k + i];
            nflip += 1;
            s |= 1 << i;
        }
    }
    if nflip % 2 != 0 {
        let mut imin = 0;
        let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
        for i in 1..8 {
            let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
            if ax < min {
                min = ax;
                imin = i;
            }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= 1 << imin;
    }
    s & 127
}

// ---------------- IQ2_XXS (66 B / 256) ----------------

pub fn enc_iq2_xxs(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid_iq2xxs();
    let k_max_q = 3i32;
    let ones = [1.0f32; 256];
    for (ibl, xbl) in x.chunks(256).enumerate() {
        let base = out.len();
        out.resize(base + 66, 0);
        let sigma2 = xbl.iter().map(|v| v * v).sum::<f32>() / 256.0;
        let mut q2 = [0u32; 16];
        let mut max_scale = 0f32;
        let mut scales = [0f32; 8];
        for ib in 0..8 {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let qw: &[f32] = im.map_or(&ones[..32], |m| &m[256 * ibl + 32 * ib..256 * ibl + 32 * ib + 32]);
            let mut weight = [0f32; 32];
            let mut waux = [0f32; 32];
            for i in 0..32 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }
            let mut xval = [0f32; 32];
            let mut block_signs = [0u8; 4];
            for k in 0..4 {
                block_signs[k] = signs_with_parity(xb, &weight, k, &mut xval);
            }
            let max = xval.iter().cloned().fold(xval[0], f32::max);
            let mut ls = [0i8; 32];
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }
            let mut lp = [0u8; 32];
            let mut scale = make_qp_quants(32, k_max_q + 1, &xval, &mut lp, &weight);
            let eff_max = scale * k_max_q as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0f32;
            let mut laux = [0i8; 32];
            for is in -6i32..=6 {
                let id = (2 * k_max_q - 1) as f32 + is as f32 * 0.1;
                let id = id / eff_max;
                let this_scale = 1.0 / id;
                for k in 0..4 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        laux[8 * k + i] = l as i8;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as usize) << (2 * i);
                    }
                    if g.map[u] < 0 {
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    ls.copy_from_slice(&laux);
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..4 {
                    let mut u = 0usize;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        u |= (l as usize) << (2 * i);
                    }
                    let grid_index = if g.map[u] >= 0 {
                        g.map[u] as usize
                    } else {
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], scale, &mut ls[8 * k..])
                    };
                    let pg = &g.grid[grid_index];
                    for i in 0..8 {
                        ls[8 * k + i] = (pg[i] - 1) / 2;
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * ls[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..4 {
                let mut u = 0usize;
                for i in 0..8 {
                    u |= (ls[8 * k + i] as usize) << (2 * i);
                }
                let grid_index = g.map[u];
                assert!(grid_index >= 0, "IQ2_XXS point not on grid");
                q2[2 * ib] |= (grid_index as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        out[base..base + 2].copy_from_slice(&f16_bytes(d));
        let id = 1.0 / d;
        for ib in 0..8 {
            let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
            q2[2 * ib + 1] |= (l as u32) << 28;
        }
        for (i, v) in q2.iter().enumerate() {
            out[base + 2 + 4 * i..base + 2 + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

pub fn dec_iq2_xxs(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(66) {
        let d = f16_val(&blk[0..2]);
        for ib32 in 0..8 {
            let a0 = u32::from_le_bytes(blk[2 + 8 * ib32..6 + 8 * ib32].try_into().unwrap());
            let a1 = u32::from_le_bytes(blk[6 + 8 * ib32..10 + 8 * ib32].try_into().unwrap());
            let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 255) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db * grid[j] as f32 * s);
                }
            }
        }
    }
}

// ---------------- IQ2_XS (74 B / 256) ----------------

pub fn enc_iq2_xs(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid_iq2xs();
    let k_max_q = 3i32;
    let ones = [1.0f32; 256];
    for (ibl, xbl) in x.chunks(256).enumerate() {
        let base = out.len();
        out.resize(base + 74, 0);
        let sigma2 = xbl.iter().map(|v| v * v).sum::<f32>() / 256.0;
        let mut q2 = [0u16; 32];
        let mut sc_bytes = [0u8; 8];
        let mut max_scale = 0f32;
        let mut scales = [0f32; 16];
        for ib in 0..16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            let qw: &[f32] = im.map_or(&ones[..16], |m| &m[256 * ibl + 16 * ib..256 * ibl + 16 * ib + 16]);
            let mut weight = [0f32; 16];
            let mut waux = [0f32; 16];
            for i in 0..16 {
                weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }
            let mut xval = [0f32; 16];
            let mut block_signs = [0u8; 2];
            for k in 0..2 {
                block_signs[k] = signs_with_parity(xb, &weight, k, &mut xval);
            }
            let max = xval.iter().cloned().fold(xval[0], f32::max);
            let mut ls = [0i8; 16];
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0f32;
            let mut scale = max / (2 * k_max_q - 1) as f32;
            let mut is_on_grid = [true; 2];
            let mut laux = [0i8; 16];
            for is in -9i32..=9 {
                let id = ((2 * k_max_q - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut is_on_grid_aux = [true; 2];
                for k in 0..2 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        laux[8 * k + i] = l as i8;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as usize) << (2 * i);
                    }
                    if g.map[u] < 0 {
                        is_on_grid_aux[k] = false;
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..16 {
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    ls.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let n_not_ongrid = is_on_grid.iter().filter(|v| !**v).count();
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        u |= (l as usize) << (2 * i);
                        ls[8 * k + i] = l as i8;
                    }
                    if g.map[u] < 0 {
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], scale, &mut ls[8 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..16 {
                    let q = (2 * ls[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs[k] = (!block_signs[k]) & 127;
                }
            }
            for k in 0..2 {
                let mut u = 0usize;
                for i in 0..8 {
                    u |= (ls[8 * k + i] as usize) << (2 * i);
                }
                let grid_index = g.map[u];
                assert!(grid_index >= 0, "IQ2_XS point not on grid");
                q2[2 * ib + k] = grid_index as u16 | ((block_signs[k] as u16) << 9);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        out[base..base + 2].copy_from_slice(&f16_bytes(d));
        let id = 1.0 / d;
        for ib in 0..16 {
            let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
            if ib % 2 == 0 {
                sc_bytes[ib / 2] = l;
            } else {
                sc_bytes[ib / 2] |= l << 4;
            }
        }
        for (i, v) in q2.iter().enumerate() {
            out[base + 2 + 2 * i..base + 4 + 2 * i].copy_from_slice(&v.to_le_bytes());
        }
        out[base + 66..base + 74].copy_from_slice(&sc_bytes);
    }
}

pub fn dec_iq2_xs(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(74) {
        let d = f16_val(&blk[0..2]);
        for ib32 in 0..8 {
            let sc = blk[66 + ib32];
            let db = [
                d * (0.5 + (sc & 0xF) as f32) * 0.25,
                d * (0.5 + (sc >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = u16::from_le_bytes(blk[2 + 8 * ib32 + 2 * l..4 + 8 * ib32 + 2 * l].try_into().unwrap());
                let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db[l / 2] * grid[j] as f32 * s);
                }
            }
        }
    }
}

// ---------------- IQ2_S (82 B / 256) ----------------

pub fn enc_iq2_s(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    let g = grid_iq2s();
    let k_max_q = 3i32;
    for (ibl, xbl) in x.chunks(256).enumerate() {
        let base = out.len();
        out.resize(base + 82, 0);
        let sigma2 = 2.0 * xbl.iter().map(|v| v * v).sum::<f32>() / 256.0;
        let mut max_scale = 0f32;
        let mut scales = [0f32; 16];
        for ib in 0..16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            let mut weight = [0f32; 16];
            let mut waux = [0f32; 16];
            for i in 0..16 {
                weight[i] = match im {
                    Some(m) => m[256 * ibl + 16 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => 0.25 * sigma2 + xb[i] * xb[i],
                };
                waux[i] = weight[i].sqrt();
            }
            let mut xval = [0f32; 16];
            let mut block_signs = [0u8; 2];
            for k in 0..2 {
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s;
            }
            let max = xval.iter().cloned().fold(xval[0], f32::max);
            let mut ls = [0i8; 16];
            if max < GROUP_MAX_EPS_IQ2_S {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0f32;
            let mut scale = max / (2 * k_max_q - 1) as f32;
            let mut is_on_grid = [true; 2];
            let mut laux = [0i8; 16];
            for is in -9i32..=9 {
                let id = ((2 * k_max_q - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut is_on_grid_aux = [true; 2];
                for k in 0..2 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        laux[8 * k + i] = l as i8;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as usize) << (2 * i);
                    }
                    if g.map[u] < 0 {
                        is_on_grid_aux[k] = false;
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], this_scale, &mut laux[8 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..16 {
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    ls.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let n_not_ongrid = is_on_grid.iter().filter(|v| !**v).count();
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u = 0usize;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        u |= (l as usize) << (2 * i);
                        ls[8 * k + i] = l as i8;
                    }
                    if g.map[u] < 0 {
                        find_best_neighbour(g, g.map[u], &xval[8 * k..], &waux[8 * k..], scale, &mut ls[8 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..16 {
                    let q = (2 * ls[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs[k] = !block_signs[k];
                }
            }
            for k in 0..2 {
                let mut u = 0usize;
                for i in 0..8 {
                    u |= (ls[8 * k + i] as usize) << (2 * i);
                }
                let grid_index = g.map[u];
                assert!(grid_index >= 0, "IQ2_S point not on grid");
                let i8v = 2 * ib + k;
                out[base + 2 + i8v] = (grid_index & 255) as u8;
                out[base + 2 + 64 + i8v / 4] |= ((grid_index >> 8) as u8) << (2 * (i8v % 4));
                out[base + 2 + 32 + i8v] = block_signs[k];
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            continue;
        }
        let d = max_scale / 31.0;
        out[base..base + 2].copy_from_slice(&f16_bytes(d * 0.9875));
        let id = 1.0 / d;
        for ib in 0..16 {
            let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
            let so = base + 2 + 64 + 8 + ib / 2;
            if ib % 2 == 0 {
                out[so] = l;
            } else {
                out[so] |= l << 4;
            }
        }
    }
}

pub fn dec_iq2_s(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(82) {
        let d = f16_val(&blk[0..2]);
        let qs = &blk[2..66]; // 32 grid-low + 32 signs
        let qh = &blk[66..74];
        let sc = &blk[74..82];
        for ib32 in 0..8 {
            let db = [
                d * (0.5 + (sc[ib32] & 0xF) as f32) * 0.25,
                d * (0.5 + (sc[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let idx = qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let signs = qs[32 + 4 * ib32 + l];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db[l / 2] * grid[j] as f32 * s);
                }
            }
        }
    }
}

// ---------------- IQ3_XXS (98 B / 256) & IQ3_S (110 B / 256) ----------------

fn enc_iq3(five: bool, x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    // five == false: IQ3_XXS (grid 256, scales+signs packed in u32)
    // five == true : IQ3_S   (grid 512, qh plane, verbatim sign bytes)
    let g = if five { grid_iq3s() } else { grid_iq3xxs() };
    let k_max_q = 8i32;
    let block_bytes = if five { 110 } else { 98 };
    let (is_lo, is_hi, is_step) = if five { (-9i32, 9i32, 0.2f32) } else { (-15, 15, 0.2) };
    for (ibl, xbl) in x.chunks(256).enumerate() {
        let base = out.len();
        out.resize(base + block_bytes, 0);
        let sigma2 = 2.0 * xbl.iter().map(|v| v * v).sum::<f32>() / 256.0;
        let mut max_scale = 0f32;
        let mut scales = [0f32; 8];
        let mut sas = [0u32; 8]; // XXS scales-and-signs words
        for ib in 0..8 {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let mut weight = [0f32; 32];
            let mut waux = [0f32; 32];
            for i in 0..32 {
                weight[i] = match im {
                    Some(m) => m[256 * ibl + 32 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
                    None => xb[i] * xb[i],
                };
                waux[i] = weight[i].sqrt();
            }
            let mut xval = [0f32; 32];
            let mut block_signs = [0u8; 4];
            for k in 0..4 {
                if five {
                    let mut s = 0u8;
                    for i in 0..8 {
                        if xb[8 * k + i] >= 0.0 {
                            xval[8 * k + i] = xb[8 * k + i];
                        } else {
                            xval[8 * k + i] = -xb[8 * k + i];
                            s |= 1 << i;
                        }
                    }
                    block_signs[k] = s;
                } else {
                    block_signs[k] = signs_with_parity(xb, &weight, k, &mut xval);
                }
            }
            let max = xval.iter().cloned().fold(xval[0], f32::max);
            let mut ls = [0i8; 32];
            let eps = if five { 0.0 } else { GROUP_MAX_EPS_IQ3_XXS };
            if max <= eps {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0f32;
            let mut scale = max / (2 * k_max_q - 1) as f32;
            let mut is_on_grid = [!five; 8]; // XXS inits true, S inits false
            let mut laux = [0i8; 32];
            for is in is_lo..=is_hi {
                let id = ((2 * k_max_q - 1) as f32 + is as f32 * is_step) / max;
                let this_scale = 1.0 / id;
                let mut is_on_grid_aux = [true; 8];
                for k in 0..8 {
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        laux[4 * k + i] = l as i8;
                    }
                    let mut u = 0usize;
                    for i in 0..4 {
                        u |= (laux[4 * k + i] as usize) << (3 * i);
                    }
                    if g.map[u] < 0 {
                        is_on_grid_aux[k] = false;
                        find_best_neighbour(g, g.map[u], &xval[4 * k..], &waux[4 * k..], this_scale, &mut laux[4 * k..]);
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    ls.copy_from_slice(&laux);
                    is_on_grid = is_on_grid_aux;
                }
            }
            let n_not_ongrid = is_on_grid.iter().filter(|v| !**v).count();
            if n_not_ongrid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    // IQ3_S re-projects every group; XXS only off-grid ones
                    if !five && is_on_grid[k] {
                        continue;
                    }
                    let mut u = 0usize;
                    for i in 0..4 {
                        let l = nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)).clamp(0, k_max_q - 1);
                        u |= (l as usize) << (3 * i);
                    }
                    let grid_index = if g.map[u] >= 0 {
                        g.map[u] as usize
                    } else {
                        find_best_neighbour(g, g.map[u], &xval[4 * k..], &waux[4 * k..], scale, &mut ls[4 * k..])
                    };
                    let pg = &g.grid[grid_index];
                    for i in 0..4 {
                        ls[4 * k + i] = (pg[i] - 1) / 2;
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * ls[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = if five { !block_signs[k] } else { (!block_signs[k]) & 127 };
                }
            }
            for k in 0..8 {
                let mut u = 0usize;
                for i in 0..4 {
                    u |= (ls[4 * k + i] as usize) << (3 * i);
                }
                let grid_index = g.map[u];
                assert!(grid_index >= 0, "IQ3 point not on grid");
                if five {
                    out[base + 2 + 8 * ib + k] = (grid_index & 255) as u8;
                    out[base + 2 + 64 + ib] |= ((grid_index >> 8) as u8) << k;
                } else {
                    out[base + 2 + 8 * ib + k] = grid_index as u8;
                }
            }
            if five {
                for k in 0..4 {
                    out[base + 2 + 64 + 8 + 4 * ib + k] = block_signs[k];
                }
            } else {
                sas[ib] = block_signs[0] as u32
                    | ((block_signs[1] as u32) << 7)
                    | ((block_signs[2] as u32) << 14)
                    | ((block_signs[3] as u32) << 21);
            }
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }
        if max_scale == 0.0 {
            // C memsets the whole quant region; bytes are already zero.
            continue;
        }
        let d = max_scale / 31.0;
        let fudge = if five { 1.033 } else { 1.0125 };
        out[base..base + 2].copy_from_slice(&f16_bytes(d * fudge));
        let id = 1.0 / d;
        if five {
            for ib in (0..8).step_by(2) {
                let l1 = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
                let l2 = nearest_int(0.5 * (id * scales[ib + 1] - 1.0)).clamp(0, 15) as u8;
                out[base + 2 + 64 + 8 + 32 + ib / 2] = l1 | (l2 << 4);
            }
        } else {
            for ib in 0..8 {
                let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
                sas[ib] |= (l as u32) << 28;
            }
            for (i, v) in sas.iter().enumerate() {
                out[base + 2 + 64 + 4 * i..base + 2 + 64 + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

pub fn enc_iq3_xxs(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    enc_iq3(false, x, im, out);
}
pub fn enc_iq3_s(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    enc_iq3(true, x, im, out);
}

pub fn dec_iq3_xxs(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(98) {
        let d = f16_val(&blk[0..2]);
        let qs = &blk[2..66];
        for ib32 in 0..8 {
            let aux = u32::from_le_bytes(blk[66 + 4 * ib32..70 + 4 * ib32].try_into().unwrap());
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g1[j] as f32 * s);
                }
                for j in 0..4 {
                    let s = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g2[j] as f32 * s);
                }
            }
        }
    }
}

pub fn dec_iq3_s(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(110) {
        let d = f16_val(&blk[0..2]);
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let signs = &blk[74..106];
        let sc = &blk[106..110];
        for ib32 in 0..8 {
            let db = if ib32 % 2 == 0 {
                d * (1 + 2 * (sc[ib32 / 2] & 0xF) as i32) as f32
            } else {
                d * (1 + 2 * (sc[ib32 / 2] >> 4) as i32) as f32
            };
            for l in 0..4 {
                let i1 = qs[8 * ib32 + 2 * l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 256);
                let i2 = qs[8 * ib32 + 2 * l + 1] as usize | (((qh[ib32] as usize) << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[i1].to_le_bytes();
                let g2 = IQ3S_GRID[i2].to_le_bytes();
                let sg = signs[4 * ib32 + l];
                for j in 0..4 {
                    let s = if sg & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g1[j] as f32 * s);
                }
                for j in 0..4 {
                    let s = if sg & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g2[j] as f32 * s);
                }
            }
        }
    }
}

// ---------------- IQ4_NL (18 B / 32) & IQ4_XS (136 B / 256) ----------------

fn best_index_int8(vals: &[i8; 16], x: f32) -> usize {
    if x <= vals[0] as f32 {
        return 0;
    }
    if x >= vals[15] as f32 {
        return 15;
    }
    let mut ml = 0;
    let mut mu = 15;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < vals[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (vals[mu - 1] as f32) < vals[mu] as f32 - x {
        mu - 1
    } else {
        mu
    }
}

/// Shared IQ4 core (ntry=7, imatrix-aware), following quantize_row_iq4_nl_impl.
fn iq4_impl(
    super_block: usize,
    x: &[f32],
    qw: Option<&[f32]>,
    d_out: &mut [u8],
    q4: &mut [u8],
    scales_h: Option<&mut [u8]>, // 2 bytes
    scales_l: Option<&mut [u8]>, // 4 bytes
) {
    let values = &KVALUES_IQ4NL;
    let ntry = 7i32;
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / super_block as f32;
    let nb = super_block / 32;
    let mut ls_all = vec![0u8; super_block];
    let mut scales = vec![0f32; nb];
    let mut max_scale = 0f32;
    let mut amax_scale = 0f32;
    for ib in 0..nb {
        let xb = &x[32 * ib..32 * ib + 32];
        let mut weight = [0f32; 32];
        for j in 0..32 {
            weight[j] = match qw {
                Some(m) => m[32 * ib + j] * (sigma2 + xb[j] * xb[j]).sqrt(),
                None => xb[j] * xb[j],
            };
        }
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let mut d = -max / values[0] as f32;
        let mut id = 1.0 / d;
        let mut sumqx = 0f32;
        let mut sumq2 = 0f32;
        for j in 0..32 {
            let l = best_index_int8(values, id * xb[j]);
            ls_all[32 * ib + j] = l as u8;
            let q = values[l] as f32;
            sumqx += weight[j] * q * xb[j];
            sumq2 += weight[j] * q * q;
        }
        d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -ntry..=ntry {
            id = (itry as f32 + values[0] as f32) / max;
            sumqx = 0.0;
            sumq2 = 0.0;
            for j in 0..32 {
                let l = best_index_int8(values, id * xb[j]);
                let q = values[l] as f32;
                sumqx += weight[j] * q * xb[j];
                sumq2 += weight[j] * q * q;
            }
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }
    if nb > 1 {
        let sh = scales_h.unwrap();
        let sl = scales_l.unwrap();
        let d = -max_scale / 32.0;
        d_out.copy_from_slice(&f16_bytes(d));
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut h16 = 0u16;
        for ib in 0..nb {
            let l = nearest_int(id * scales[ib]).clamp(-32, 31);
            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let xb = &x[32 * ib..32 * ib + 32];
            for j in 0..32 {
                ls_all[32 * ib + j] = best_index_int8(values, idl * xb[j]) as u8;
            }
            let lu = (l + 32) as u8;
            if ib % 2 == 0 {
                sl[ib / 2] = lu & 0xF;
            } else {
                sl[ib / 2] |= (lu & 0xF) << 4;
            }
            h16 |= ((lu >> 4) as u16) << (2 * ib);
        }
        sh.copy_from_slice(&h16.to_le_bytes());
    } else {
        d_out.copy_from_slice(&f16_bytes(scales[0]));
        let id = if scales[0] != 0.0 { 1.0 / scales[0] } else { 0.0 };
        for j in 0..super_block {
            ls_all[j] = best_index_int8(values, id * x[j]) as u8;
        }
    }
    for i in 0..super_block / 32 {
        for j in 0..16 {
            q4[16 * i + j] = ls_all[32 * i + j] | (ls_all[32 * i + 16 + j] << 4);
        }
    }
}

pub fn enc_iq4_nl(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    for (ib, blk) in x.chunks(32).enumerate() {
        let base = out.len();
        out.resize(base + 18, 0);
        let qw = im.map(|m| &m[32 * ib..32 * ib + 32]);
        let (d, rest) = out[base..].split_at_mut(2);
        iq4_impl(32, blk, qw, d, &mut rest[..16], None, None);
    }
}

pub fn enc_iq4_xs(x: &[f32], im: Option<&[f32]>, out: &mut Vec<u8>) {
    for (ibl, blk) in x.chunks(256).enumerate() {
        let base = out.len();
        out.resize(base + 136, 0);
        let qw = im.map(|m| &m[256 * ibl..256 * ibl + 256]);
        let (head, qs) = out[base..base + 136].split_at_mut(8);
        let (d, head) = head.split_at_mut(2);
        let (sh, sl) = head.split_at_mut(2);
        iq4_impl(256, blk, qw, d, qs, Some(sh), Some(sl));
    }
}

pub fn dec_iq4_nl(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(18) {
        let d = f16_val(&blk[0..2]);
        let qs = &blk[2..18];
        for j in 0..16 {
            out.push(d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32);
        }
        for j in 0..16 {
            out.push(d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32);
        }
    }
}

pub fn dec_iq4_xs(data: &[u8], out: &mut Vec<f32>) {
    for blk in data.chunks(136) {
        let d = f16_val(&blk[0..2]);
        let sh = u16::from_le_bytes(blk[2..4].try_into().unwrap());
        let sl = &blk[4..8];
        let qs = &blk[8..136];
        for ib in 0..8 {
            let ls = (((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32) | ((((sh >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            for j in 0..16 {
                out.push(dl * KVALUES_IQ4NL[(qs[16 * ib + j] & 0xF) as usize] as f32);
            }
            for j in 0..16 {
                out.push(dl * KVALUES_IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32 - 0.5) * 4.0
            })
            .collect()
    }

    fn rt(name: &str, enc: fn(&[f32], Option<&[f32]>, &mut Vec<u8>), dec: fn(&[u8], &mut Vec<f32>), bytes_per_256: usize, tol: f32) {
        let x = test_vec(512, 0x1234_5678);
        let im = vec![1.0f32; 512];
        let mut e = Vec::new();
        enc(&x, Some(&im), &mut e);
        assert_eq!(e.len(), bytes_per_256 * 2, "{name} size");
        let mut d = Vec::new();
        dec(&e, &mut d);
        assert_eq!(d.len(), 512, "{name} decode len");
        let rmse = (x.iter().zip(&d).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 512.0).sqrt();
        let xrms = (x.iter().map(|a| a * a).sum::<f32>() / 512.0).sqrt();
        assert!(rmse / xrms < tol, "{name} relative rmse {} exceeds {}", rmse / xrms, tol);
    }

    #[test]
    fn iq_roundtrips() {
        rt("iq4_xs", enc_iq4_xs, dec_iq4_xs, 136, 0.08);
        rt("iq3_s", enc_iq3_s, dec_iq3_s, 110, 0.25);
        rt("iq3_xxs", enc_iq3_xxs, dec_iq3_xxs, 98, 0.35);
        rt("iq2_s", enc_iq2_s, dec_iq2_s, 82, 0.62);
        rt("iq2_xs", enc_iq2_xs, dec_iq2_xs, 74, 0.65);
        rt("iq2_xxs", enc_iq2_xxs, dec_iq2_xxs, 66, 0.75);
    }

    #[test]
    fn iq4_nl_roundtrip() {
        let x = test_vec(64, 0xdead_beef);
        let im = vec![1.0f32; 64];
        let mut e = Vec::new();
        enc_iq4_nl(&x, Some(&im), &mut e);
        assert_eq!(e.len(), 36);
        let mut d = Vec::new();
        dec_iq4_nl(&e, &mut d);
        assert_eq!(d.len(), 64);
        let rmse = (x.iter().zip(&d).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / 64.0).sqrt();
        assert!(rmse < 0.1, "iq4_nl rmse {rmse}");
    }

    #[test]
    fn grids_build() {
        assert_eq!(grid_iq2xxs().grid.len(), 256);
        assert_eq!(grid_iq2xs().grid.len(), 512);
        assert_eq!(grid_iq2s().grid.len(), 1024);
        assert_eq!(grid_iq3xxs().grid.len(), 256);
        assert_eq!(grid_iq3s().grid.len(), 512);
        // every on-grid index maps to itself
        let g = grid_iq2xxs();
        let on: usize = g.map.iter().filter(|v| **v >= 0).count();
        assert_eq!(on, 256);
    }
}
