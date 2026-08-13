//! Exact-fit quant-mix solver: multiple-choice knapsack via Lagrangian
//! relaxation, then greedy upgrades to spend remaining slack.

use crate::gguf::GgmlType;

pub struct Candidate {
    pub ty: GgmlType,
    pub bytes: u64,
    /// imatrix-weighted squared error of quantizing the tensor to this type
    pub err: f64,
}

pub struct TensorChoices {
    pub tensor_idx: usize,
    /// sorted by bytes ascending
    pub cands: Vec<Candidate>,
}

/// Returns chosen candidate index per TensorChoices entry, or None if even the
/// smallest mix exceeds the budget.
pub fn solve(tensors: &[TensorChoices], budget: u64) -> Option<Vec<usize>> {
    let min_total: u64 = tensors.iter().map(|t| t.cands[0].bytes).sum();
    if min_total > budget {
        return None;
    }

    let pick = |lambda: f64| -> (Vec<usize>, u64) {
        let mut total = 0u64;
        let mut sel = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mut best = 0usize;
            let mut best_cost = f64::INFINITY;
            for (i, c) in t.cands.iter().enumerate() {
                let cost = c.err + lambda * c.bytes as f64;
                if cost < best_cost {
                    best_cost = cost;
                    best = i;
                }
            }
            total += t.cands[best].bytes;
            sel.push(best);
        }
        (sel, total)
    };

    // lambda = 0 picks the lowest-error (largest) mix; if it fits, done —
    // greedy pass below can't improve on the max-quality mix.
    let (sel, total) = pick(0.0);
    let mut best_fit = if total <= budget {
        (sel, total)
    } else {
        // bisect lambda until the mix fits snugly
        let mut lo = 0.0f64; // too loose (over budget)
        let mut hi = 1.0f64;
        while pick(hi).1 > budget {
            hi *= 8.0;
            if hi > 1e30 {
                return None; // cannot happen given min_total check, but be safe
            }
        }
        let mut best = pick(hi);
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let (sel, total) = pick(mid);
            if total <= budget {
                hi = mid;
                best = (sel, total);
            } else {
                lo = mid;
            }
        }
        best
    };

    // Greedy top-up: spend remaining slack on the best err-reduction per byte.
    loop {
        let slack = budget - best_fit.1;
        let mut best_gain = 0f64;
        let mut best_move: Option<(usize, usize)> = None;
        for (ti, t) in tensors.iter().enumerate() {
            let cur = best_fit.0[ti];
            let cur_c = &t.cands[cur];
            for (ci, c) in t.cands.iter().enumerate() {
                if c.bytes <= cur_c.bytes || c.err >= cur_c.err {
                    continue;
                }
                let extra = c.bytes - cur_c.bytes;
                if extra > slack {
                    continue;
                }
                let gain = (cur_c.err - c.err) / extra as f64;
                if gain > best_gain {
                    best_gain = gain;
                    best_move = Some((ti, ci));
                }
            }
        }
        match best_move {
            Some((ti, ci)) => {
                best_fit.1 += tensors[ti].cands[ci].bytes - tensors[ti].cands[best_fit.0[ti]].bytes;
                best_fit.0[ti] = ci;
            }
            None => break,
        }
    }

    Some(best_fit.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    fn tc(idx: usize, opts: &[(u64, f64)]) -> TensorChoices {
        TensorChoices {
            tensor_idx: idx,
            cands: opts
                .iter()
                .map(|&(bytes, err)| Candidate { ty: GgmlType::Q4_0, bytes, err })
                .collect(),
        }
    }

    #[test]
    fn picks_max_quality_when_it_fits() {
        let ts = vec![tc(0, &[(10, 5.0), (20, 1.0)]), tc(1, &[(10, 5.0), (20, 1.0)])];
        let sel = solve(&ts, 100).unwrap();
        assert_eq!(sel, vec![1, 1]);
    }

    #[test]
    fn respects_budget() {
        let ts = vec![tc(0, &[(10, 5.0), (20, 1.0)]), tc(1, &[(10, 5.0), (20, 4.9)])];
        let sel = solve(&ts, 30).unwrap();
        // one upgrade affordable; tensor 0 has the better gain
        assert_eq!(sel, vec![1, 0]);
    }

    #[test]
    fn infeasible() {
        let ts = vec![tc(0, &[(10, 5.0)])];
        assert!(solve(&ts, 5).is_none());
    }
}
