# shoehorn — Design Notes

A running log of every significant choice and why it was made. Newest sections
are appended at the bottom.

## What this is

`shoehorn` takes a BF16 GGUF model + an importance matrix (imatrix) and produces
a mixed-precision quantized GGUF whose size lands as close as possible to (and
never over) the memory actually available to the GPU, then runs it with
llama.cpp. The user request verbatim: *"LLM runtime that takes in a BF16 model
file and an imatrix and quantizes it to fit exactly into your available VRAM.
Can you make that?"*

## D1. Own quantizer, llama.cpp for inference

**Choice** (user-selected from options): implement the quantization kernels
ourselves — GGUF in, GGUF out — but reuse llama.cpp as the inference engine.

**Why**: writing a Metal inference engine is a months-long project and mostly
commodity work; the interesting novel part is the *exact-fit solver* and the
imatrix-weighted encoders. Emitting standard GGUF means the output is usable by
the whole llama.cpp ecosystem, and llama.cpp doubles as an independent
correctness oracle: if our encoders are wrong, the model produces garbage text.

**Target** (user-selected): Apple Silicon / Metal first. "VRAM" on unified
memory means Metal's `recommendedMaxWorkingSetSize` (the OS's answer for how
much the GPU may reasonably wire), not total RAM.

## D2. Language: Rust

Single fast binary, rayon for data-parallel quantization over tensors, `metal`
crate for the working-set probe, `memmap2` so the BF16 source is paged in
lazily rather than loaded. Python+numpy could express the math but would be
painfully slow for the per-block scale searches (each 32-element block tries
~19 candidate grids).

## D3. Quant format lineup

Implemented: **Q4_0, Q4_1, Q5_0, Q5_1, Q8_0** (32-element blocks) and
**Q4_K, Q5_K, Q6_K** (256-element super-blocks), plus F16/BF16/F32 passthrough.

**Why these**: they span ~4.5 to 16 bits/weight, giving the solver a dense
quality ladder. K-quants are the workhorses (best quality per bit of the
non-IQ formats and imatrix-aware); the legacy 32-block formats serve as
fallbacks for tensors whose row length isn't divisible by 256 (K-quants
require that). IQ formats (IQ2/IQ3/IQ4) were left out of v1: their codebook
search is an order of magnitude more code and they mostly matter below
4 bpw — noted as a future extension, the solver design accommodates adding
candidates trivially.

**Compatibility rule**: a tensor can only use a type whose block size divides
its row length (ne0 % 256 for K-quants, % 32 for the rest). 1D tensors
(norms) and biases stay F32, matching llama.cpp convention — they're tiny and
numerically sensitive.

## D4. Weighted quantization math mirrors ggml

The encoders reimplement the semantics of ggml's reference quantizers:

- `make_qx_quants` — symmetric formats (Q4_0/Q5_0/Q6_K sub-scales): weighted
  least-squares fit of scale `d` over ~19 candidate grids
  (`iscale = -(nmax + 0.1·is)/max`, is ∈ [-9, 9]), keeping the grid maximizing
  `(Σw·x·l)²/Σw·l²`.
- `make_qkx3_quants` — asymmetric formats (Q4_1/Q5_1/Q4_K/Q5_K sub-blocks):
  weighted linear regression solving scale *and* min per candidate grid
  (rmin=-0.9, rdelta=0.05, 36 steps), clamping min ≥ 0 in the K-quant "d·q −
  dmin·m" convention.
- Element weight: `w[j] = imatrix[j] · sqrt(σ² + x[j]²)` with σ² the row's
  mean square — same shaping ggml's `quantize_row_*_impl` uses. Without an
  imatrix entry the weight degrades to `sqrt(σ² + x²)` (ggml's fallback), so
  tensors missing from the imatrix (typically `token_embd`) still quantize
  sensibly.

**Why mirror ggml instead of inventing our own objective**: the *decoder* is
fixed (llama.cpp's dequant kernels), so the encoder's job is to pick the best
representable points under that decoder — a solved problem whose reference
solution is battle-tested. Novelty budget is spent on the solver instead.

**Verification**: every format has a decoder in-crate used for (a) unit-test
round-trips against tolerance, (b) the solver's error measurements, and (c)
cross-checking against llama.cpp by loading the output model.

## D5. Error metric for the solver

Per tensor per candidate type: full encode+decode round trip, error
`Σ imatrix[j]·(x[j]−x̂[j])²` summed over all elements. This is the true
end-to-end distortion under the importance weighting, not a proxy — affordable
because quantization is embarrassingly parallel across rows and run once.

The imatrix weight is applied per *column* (position within a row), because
imatrix entries are accumulated activation second moments of the columns each
weight multiplies — exactly llama.cpp's interpretation.

## D6. Exact-fit solving: Lagrangian knapsack + greedy top-up

Multiple-choice knapsack: per tensor pick one type from its candidate set,
minimize total weighted error s.t. total bytes ≤ budget.

1. Bisect the Lagrange multiplier λ; per tensor pick
   `argmin(err + λ·bytes)` — 30 iterations converge far below one byte of
   slack in practice.
2. Greedy refinement on the remaining slack: repeatedly take the single-tensor
   upgrade with the best Δerr/Δbytes that still fits. This spends the last few
   MB the relaxation leaves on the table, honoring "fit *exactly*".

**Why not DP**: byte-resolution DP over multi-GB budgets is infeasible;
Lagrangian relaxation + top-up is the standard practical solution and gets
within a rounding error of the frontier.

## D7. The budget model

```
model_budget = usable_vram − kv_cache − compute_buffer − reserve
```

- `usable_vram`: Metal `recommendedMaxWorkingSetSize` (≈75% of unified RAM),
  overridable with `--budget` (so you can target a *different* machine or an
  artificial size).
- `kv_cache`: exact: `n_layer · ctx · n_kv_heads · (key_len + value_len) · 2`
  bytes (f16 KV), from the model's own GGUF hyperparams and `--ctx`.
- `compute_buffer`: estimate: logits (`ubatch·vocab·4`) + activation scratch
  (`ubatch·embd·4·8`) — deliberately rough; the `--reserve` margin (default
  512 MiB) absorbs estimate error, Metal shader buffers, and the host process.
  Documented as heuristic; measured numbers beat it, hence the override flags.

## D8. Memory strategy: measure-then-requantize

The solver pass measures errors and immediately discards encoded buffers; after
solving, chosen types are re-encoded during the streaming write. This doubles
encode compute but keeps peak memory at ~one tensor per rayon worker instead of
(all tensors × all candidate types), which matters for real (30B+) models. The
BF16 source stays mmap'd throughout.

## D9. Test model & imatrix

Qwen3-0.6B BF16 (unsloth GGUF build) — small enough for fast e2e iterations,
modern enough to exercise Qwen3 metadata paths. Imatrix generated locally with
`llama-imatrix` over ~1000 lines of prose+code calibration text; both the new
GGUF-based imatrix format (`*.in_sum2`/`*.counts` tensors) and the legacy
binary format are parsed, since most imatrices in the wild are still legacy.

## D10. Error sampling (128 rows/tensor)

Measuring true error on every row × every candidate would quintuple the
quantization work. Rows within a tensor are statistically homogeneous, so the
measurement pass scores ≤128 evenly-spaced rows per tensor and scales the sum
by rows/sampled. `--exact-errors` disables sampling. On Qwen3-0.6B the full
plan+solve takes 0.7 s wall (measured); the final write re-encodes all rows.

## Validation results (2026-08-13, M4 Pro 24 GB)

- Probe: `recommendedMaxWorkingSetSize` = 17.76 GiB of 24 GB — confirming
  "available VRAM" ≠ RAM even on unified memory.
- Forced 1.75 GiB envelope (ctx 4096 → 519.2 MiB weight budget): solver used
  **99.981%** of budget (103 KB slack), mix Q4_K/Q5_K/Q6_K/Q8_0/F16 at
  7.306 bpw overall. Choices are intuitively right: `ffn_down` (known
  quant-tolerant) got Q4_K, attention Q6_K, sensitive tensors F16.
- llama.cpp b10360 loads the file and generates coherently on Metal
  (~210 tok/s CLI, 180 tok/s server) — independent confirmation all eight
  hand-written block encoders are bit-compatible.
- Held-out perplexity (git-rebase man page, unseen by the imatrix):
  BF16 14.533 ±0.552 vs shoehorn 14.623 ±0.556 — **+0.6% PPL at 46% of the
  size**, within one sigma.
- Default run against real VRAM: model fits at F16, solver picks all-F16 and
  reports 6.9% budget use — correctly refuses to degrade when there's room.

### Gotchas hit along the way

- Recent llama-cli defaults into conversation/interactive mode even with
  `-no-cnv` + a closed stdin, spinning on `> ` prompts forever; use `-st`
  (single-turn) for scripted smoke tests, or llama-perplexity which exits.
- unsloth HF repos: the BF16 GGUF lives at the repo root, not under a `BF16/`
  subfolder — a wrong guess downloads a 15-byte "Entry not found" body that
  still exits 0; always verify magic bytes after download.
- The `metal` crate pulls the deprecated `block` crate (future-incompat
  warning) — harmless today, worth swapping for `objc2-metal` eventually.
