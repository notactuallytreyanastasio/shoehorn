# shoehorn

Quantize a BF16 GGUF model with an imatrix so it fits exactly into your
available VRAM, then run it with llama.cpp.

Preset quantizations (`Q4_K_M`, `Q5_K_S`, ...) ignore your hardware. Pick one
that fits your machine and you either leave hundreds of megabytes of quality
unused or find out at load time that it didn't fit after all. shoehorn starts
from the memory you actually have, subtracts what inference itself will need
(KV cache, compute buffers), and solves a per-tensor mixed-precision
assignment whose total size lands within a rounding error of the remainder.
Every spare megabyte goes where the importance matrix says it buys the most
model quality.

```
$ shoehorn vram
Apple M4 Pro: 17.76 GiB usable for GPU working set

$ shoehorn quantize -m Qwen3-0.6B-BF16.gguf -i qwen3.imatrix \
    --ctx 4096 --budget 1.75GiB -o fitted.gguf
...
weights: 519.2 MiB of 519.2 MiB budget (99.981% used, 103424 B slack) | overall 7.306 bpw

$ shoehorn run -m fitted.gguf --ctx 4096
```

The quantizer is implemented from scratch in this repo (Rust, no llama.cpp
code linked). The output is standard GGUF v3 that any llama.cpp build, or
anything downstream of it, loads directly. llama.cpp handles inference and
doubles as an independent correctness oracle.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [The math](#the-math)
- [Supported formats](#supported-formats)
- [The budget model](#the-budget-model)
- [CLI reference](#cli-reference)
- [Imatrix files](#imatrix-files)
- [Results](#results)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Troubleshooting](#troubleshooting)

---

## Quick start

Prerequisites: a Rust toolchain, and llama.cpp (`brew install llama.cpp`) for
generating imatrices and for `shoehorn run`.

```sh
cargo build --release

# 1. Get a BF16 (or F16/F32) GGUF of your model.
#    Most HF quant repos (unsloth, bartowski, ggml-org) publish one.

# 2. Generate an importance matrix over calibration text:
llama-imatrix -m model-bf16.gguf -f calibration.txt -o model.imatrix -ngl 99

# 3. Solve and quantize to your machine's actual capacity:
shoehorn quantize -m model-bf16.gguf -i model.imatrix --ctx 8192 -o fitted.gguf

# 4. Serve it (execs llama-server with full GPU offload):
shoehorn run -m fitted.gguf --ctx 8192
```

`shoehorn plan` takes the same flags as `quantize` without `-o` and prints the
solved per-tensor mix without writing anything, so you can preview what a
budget implies before spending the encode time.

## How it works

The pipeline runs in five stages.

Probing comes first. On Apple Silicon, "VRAM" is not RAM: Metal will only wire
a fraction of unified memory for the GPU. shoehorn asks the Metal device for
`recommendedMaxWorkingSetSize` (17.76 GiB on a 24 GB M4 Pro, about 75%).
`--budget` overrides the probe, which also lets you quantize for a different
machine ("make this fit my friend's 8 GiB M1") or for an artificial envelope.

Next it computes the budget. From the target context length and the model's
own GGUF hyperparameters, shoehorn computes the KV cache size exactly and
estimates the compute buffer, then subtracts both plus a safety `--reserve`.
What remains is the weight budget. See [The budget model](#the-budget-model).

Then it measures. Every quantizable tensor gets a candidate ladder (Q4_K,
Q5_K, Q6_K, Q8_0, F16, or the legacy 32-block formats when the row length
isn't divisible by 256). Each (tensor, candidate) pair is scored by actually
encoding and decoding a sample of rows and accumulating the imatrix-weighted
squared error. That is the true end-to-end distortion under the decoder
llama.cpp will use. The work parallelizes across all cores; on Qwen3-0.6B the
whole measure and solve pass takes 0.7 s.

The solve itself is a multiple-choice knapsack: pick one type per tensor,
minimize total weighted error subject to total bytes staying at or under
budget. shoehorn uses Lagrangian relaxation (bisect the shadow price of a
byte; each tensor independently picks the candidate minimizing
`err + λ·bytes`), then a greedy pass that spends the slack the relaxation
leaves behind: repeatedly apply the single-tensor upgrade with the best
error reduction per byte that still fits. Utilization in practice exceeds
99.9% of the weight budget.

Finally it writes. Chosen types are re-encoded row-parallel and streamed out
as a GGUF v3 with all source metadata preserved and `general.file_type` set to
the dominant type for display purposes. Norms, biases, and anything else 1D
stay F32, matching llama.cpp convention: they are tiny and numerically
sensitive.

The BF16 source is mmap'd and never fully materialized. Peak memory is
roughly one tensor per worker thread, so 30B-class models are fine on a
laptop.

## The math

Quantized formats represent a block of weights as one or two f16 scale
factors plus low-bit integers: symmetric formats decode as `x̂ = d·q`,
asymmetric ones as `x̂ = d·q − m`, with sub-block scales in the K-quants. The
decoder is fixed (it's llama.cpp's dequantization), so the encoder's entire
job is choosing `d`, `m`, and the integers to minimize error under a
weighting that reflects how much each weight matters.

`llama-imatrix` supplies that weighting. It runs the model over calibration
text and accumulates, for each matmul weight, the mean squared activation of
each input column. Columns that see large activations amplify their weights'
quantization error in the layer's output, so they deserve more of the bit
budget. shoehorn uses the element weight

```
w[j] = imatrix[j] · sqrt(σ² + x[j]²)      σ² = mean square of the row
```

which is the same shaping ggml's imatrix-aware quantizers use. Tensors
without an imatrix entry (typically `token_embd`, which is never a matmul
input) fall back to the activation-agnostic `sqrt(σ² + x²)`.

For symmetric formats the encoder mirrors ggml's `make_qx_quants`: try 19
candidate grids `iscale = −(nmax + 0.1·is)/max` for is in [−9, 9]; for each,
round every element and evaluate the weighted least-squares objective; keep
the grid maximizing `(Σ w·x·l)² / Σ w·l²`, whose optimal scale is
`Σ w·x·l / Σ w·l²`. For asymmetric formats it mirrors `make_qkx3_quants`:
over 37 candidate grids, solve the two-parameter weighted regression for
scale and offset jointly, clamping the offset so the K-quant `d·q − dmin·m`
convention keeps its positive-min invariant. K-quant super-blocks then
quantize the 8 or 16 sub-block scales themselves to 6 or 8 bits and re-round
every element against the quantized scales.

Why copy ggml's objective instead of inventing one? The reference quantizers
have years of use against this exact decoder, and matching their semantics
makes shoehorn's output directly comparable to `llama-quantize`'s. The new
work went into the solver.

The solver minimizes the sum over tensors of measured weighted error, with no
per-tensor normalization: the imatrix magnitudes already encode relative
importance across tensors, and total weighted distortion is exactly what the
knapsack should minimize.

## Supported formats

| format | bits/weight | block | encode notes |
|---|---|---|---|
| Q4_K | 4.50 | 256 | 8×32 sub-blocks, 6-bit scales+mins, weighted 2-param regression |
| Q5_K | 5.50 | 256 | as Q4_K plus a high-bit plane |
| Q6_K | 6.56 | 256 | 16×16 sub-blocks, 8-bit signed scales, weighted grid search |
| Q8_0 | 8.50 | 32 | absmax scaling (error is negligible; search unnecessary) |
| Q4_0 / Q4_1 | 4.50 / 5.00 | 32 | legacy fallback for rows not divisible by 256 |
| Q5_0 / Q5_1 | 5.50 / 6.00 | 32 | legacy fallback, high-bit plane |
| F16 / BF16 / F32 | 16 / 16 / 32 | n/a | passthrough / conversion |

Rows divisible by 256 get the candidate ladder {Q4_K, Q5_K, Q6_K, Q8_0, F16};
rows divisible only by 32 get {Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16}. Every
format has a matching in-crate decoder used for error measurement and tests.

## The budget model

```
weight_budget = usable_vram − kv_cache − compute_est − reserve

kv_cache    = n_layer · ctx · n_kv_heads · (key_len + value_len) · 2   (f16 K+V, exact)
compute_est = ubatch·n_vocab·4 + ubatch·n_embd·32                      (heuristic)
ubatch      = min(512, ctx)
```

All hyperparameters come from the model's own GGUF metadata (`block_count`,
`attention.head_count_kv`, `attention.key_length`, ...), so grouped-query
attention and unusual head sizes are handled per model rather than assumed.

Worked example, Qwen3-0.6B at `--budget 1.75GiB --ctx 4096`:

```
1.75 GiB − 448 MiB KV (28 layers · 4096 · 8 kv-heads · 256 · 2 B)
         − 313 MiB compute est (512·151936·4 logits dominate)
         − 512 MiB reserve
         = 519.2 MiB for weights → solver fills 519.1 MiB of it
```

The KV term is exact. The compute term is deliberately rough, since
llama.cpp's graph allocation depends on flash-attention availability, batch
shape, and version; the `--reserve` margin absorbs its error plus Metal
shader buffers and the host process. If you have a measured number for your
setup, `--budget` and `--reserve` let you dial it in precisely.

## CLI reference

```
shoehorn plan      -m <bf16.gguf> [-i <imatrix>] [fit flags]
shoehorn quantize  -m <bf16.gguf> [-i <imatrix>] [fit flags] -o <out.gguf>
shoehorn run       -m <model.gguf> [--ctx N] [-- <llama-server args...>]
shoehorn vram
```

Fit flags, shared by `plan` and `quantize`:

| flag | default | meaning |
|---|---|---|
| `-m, --model` | required | BF16/F16/F32 source GGUF (already-quantized sources also read, via the in-crate decoders) |
| `-i, --imatrix` | none | imatrix file, legacy binary or GGUF-based; omitting it warns and falls back to activation-agnostic weighting |
| `--ctx` | 8192 | context length the KV budget is computed for |
| `--budget` | Metal probe | total memory envelope: `18GiB`, `800MB`, `4.5G`, or plain bytes |
| `--reserve` | 512MiB | safety margin subtracted from the envelope |
| `--exact-errors` | off | score every row instead of a 128-row sample per tensor |

`plan` and `quantize` print the full per-tensor table (shape, chosen type,
size, bits/weight), a by-type rollup, budget utilization with the residual
slack in bytes, and the projected total VRAM picture at the target context.

`run` execs `llama-server -m <model> -c <ctx> -ngl 99`; everything after `--`
is passed through (`--port`, `--api-key`, ...).

`vram` prints the detected device and its recommended working-set size.

## Imatrix files

Both formats found in the wild are supported and auto-detected:

- GGUF-based (current `llama-imatrix` output): tensors named
  `<name>.in_sum2` (per-column sums of squared activations) and
  `<name>.counts`; shoehorn divides one by the other.
- Legacy binary (most published imatrices): `n_entries`, then per entry
  name / ncall / nval / f32 values; values are divided by ncall.

Weights are sanitized (non-finite and non-positive entries floored) so a
degenerate imatrix can't zero out the fit. For 3D expert tensors (MoE), an
imatrix covering `ne0 × n_expert` is sliced per expert; one covering only
`ne0` is broadcast.

Calibration text matters less than having an imatrix at all. A few hundred KB
of mixed prose and code is the community norm. The test suite here uses
concatenated man pages and holds out different text for evaluation.

## Results

Qwen3-0.6B (596 M params), M4 Pro 24 GB, llama.cpp b10360, 2026-08-13.

Forced into a 1.75 GiB total envelope (weight budget 519.2 MiB, ctx 4096):

| | BF16 | shoehorn mix |
|---|---|---|
| weights on disk | 1.13 GiB | 524.8 MiB (46%) |
| bits/weight | 16 | 7.306 |
| budget utilization | n/a | **99.981%** (103 KB slack) |
| held-out perplexity¹ | 14.533 ± 0.552 | 14.623 ± 0.556 (**+0.6%**) |
| generation (llama-cli) | n/a | ~210 tok/s |

¹ git-rebase man page, text the imatrix never saw.

The solved mix matches what a llama.cpp veteran would hand-tune: `ffn_down`
(known quant-tolerant) at Q4_K, attention projections at Q6_K, Q8_0 and F16
reserved for the tensors whose weighted error per byte is worst. Here it
falls out of the optimization, per model, with no hand rules.

Against the machine's real 17.76 GiB budget, the 0.6B model fits at F16 and
the solver keeps everything at F16 (6.9% utilization). It degrades nothing
when there is no need.

## Project layout

```
src/gguf.rs      GGUF v3 reader/writer (arbitrary KVs preserved, aligned offsets)
src/quant.rs     the 8 block encoders + decoders, weighted scale search
src/imatrix.rs   legacy + GGUF imatrix parsing, weight sanitization
src/solver.rs    Lagrangian knapsack + greedy top-up
src/vram.rs      Metal working-set probe
src/main.rs      CLI, budget model, measurement orchestration (rayon)
DESIGN.md        the how/why of every decision, in order (D1-D10), gotchas
docs/            decision-graph export (deciduous)
```

## Testing

`cargo test` covers round-trip encode/decode for every format against RMSE
tolerance, a check that imatrix weighting actually shifts the fit toward
important columns, and solver unit tests (max quality when everything fits,
budget respected, infeasibility detected).

The end-to-end oracle is llama.cpp itself: an independent implementation
loads the quantized file on Metal. A single mispacked bit plane produces
garbage text, so coherent greedy-decoded output plus near-baseline held-out
perplexity is strong evidence the encoders are bit-compatible.

## Limitations and roadmap

- No formats below 4.5 bits/weight. The IQ family (IQ2/IQ3/IQ4,
  codebook-based) isn't implemented, so if a model doesn't fit at
  Q4_K-everywhere, the honest answer is a smaller model. The solver takes new
  candidates trivially; implementing IQ encoders is the extension that would
  pay off most.
- The compute buffer is an estimate. Real allocation varies by llama.cpp
  version and flash-attention path; `--reserve` absorbs the difference. A
  `--calibrate` mode that launches llama.cpp once and reads back actual
  allocations would replace the guess with a measurement.
- The probe is Metal-only. `--budget` works anywhere; a CUDA/NVML probe is
  straightforward if a Linux target materializes.
- Embedded imatrix stats, attention-sink tensors, and other exotic GGUF
  extras are passed through untouched but not exploited.

## Design history

Every significant decision, and several dead ends (a llama-cli
interactive-mode hang, a 15-byte "Entry not found" model download that exits
0), is written up as it happened in [DESIGN.md](DESIGN.md).

## Troubleshooting

- `no room for weights`: your `--ctx` KV cache plus reserve exceeds the
  envelope. Lower `--ctx`, or lower `--reserve` if you've measured real
  usage.
- `even the smallest mix exceeds the weight budget`: the model can't fit at
  ~4.5 bits/weight. Use a smaller model (see the IQ limitation above).
- Scripted smoke tests hang in llama-cli: recent llama.cpp defaults into
  conversation mode and spins on a closed stdin even with `-no-cnv`; use
  `-st` (single-turn), or `llama-perplexity`, which always exits.
- Output loads but talks nonsense: that is the oracle failing. File a bug
  with the tensor table from `shoehorn plan`; a specific format's packing is
  suspect.
