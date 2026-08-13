# shoehorn demo

```
./demo/run.sh
```

One command, a few minutes on an M-series Mac. It builds shoehorn, downloads
Qwen3-0.6B in BF16 (1.2 GB, first run only), generates an imatrix from man-page
calibration text, then quantizes the model into three memory envelopes and
validates each one in llama.cpp against held-out text the imatrix never saw.

Expected shape of the result (M4 Pro numbers):

```
BF16 baseline: 1.1G   PPL 14.53
1.75GiB  ->  525M  PPL 14.62     (7.3 bpw: Q4_K..F16 mix)
1.53GiB  ->  300M  PPL 21.53     (~4 bpw: IQ3/IQ4/Q5_K mix)
1.44GiB  ->  207M  PPL 212.7     (2.8 bpw: IQ2-heavy mix)
```

Each file lands within ~0.1% of its byte budget. The bottom row is
deliberately brutal — sub-3 bpw wrecks a 0.6B model no matter who quantizes
it — but the same mix beats `llama-quantize IQ2_XXS` (PPL 446.8 at a *larger*
file) because the solver allocates bits per tensor instead of uniformly.

For a bigger-model run (Qwen3-14B against a real 17.76 GiB Apple Silicon
budget, and squeezed into 8 GiB), see the Results section of the main README.
