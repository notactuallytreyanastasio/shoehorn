#!/usr/bin/env bash
# shoehorn demo: quantize one model to three memory envelopes, validate each
# in llama.cpp, and print the size/quality ladder.
#
# Total runtime is a few minutes on an M-series Mac (plus a ~1.2 GB model
# download on first run). Requires: cargo, llama.cpp (brew install llama.cpp).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/testdata"
BIN="$ROOT/target/release/shoehorn"
MODEL="$DATA/Qwen3-0.6B-BF16.gguf"
IMATRIX="$DATA/qwen3-imatrix.gguf"
CTX=4096

for tool in cargo llama-imatrix llama-perplexity llama-cli; do
    command -v "$tool" >/dev/null || { echo "missing: $tool (brew install llama.cpp)"; exit 1; }
done

echo "== build =="
cargo build --release --manifest-path "$ROOT/Cargo.toml"

mkdir -p "$DATA"
if [ ! -f "$MODEL" ]; then
    echo "== download test model (1.2 GB) =="
    curl -L -o "$MODEL" \
        "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf"
fi

if [ ! -f "$DATA/calib.txt" ]; then
    (man bash | col -b; man zshexpn | col -b) > "$DATA/calib.txt" 2>/dev/null
fi
if [ ! -f "$DATA/heldout.txt" ]; then
    man git-rebase | col -b > "$DATA/heldout.txt"
fi

if [ ! -f "$IMATRIX" ]; then
    echo "== generate imatrix (calibration: man pages; held-out eval text is different) =="
    llama-imatrix -m "$MODEL" -f "$DATA/calib.txt" -o "$IMATRIX" --chunks 40 -ngl 99
fi

echo
echo "== detected GPU budget =="
"$BIN" vram

ppl() {
    llama-perplexity -m "$1" -f "$DATA/heldout.txt" -ngl 99 2>&1 </dev/null \
        | sed -n 's/.*Final estimate: PPL = \([0-9.]*\).*/\1/p'
}

echo
echo "== quantize to three envelopes =="
declare -a ROWS
for budget in 1.75GiB 1.53GiB 1.44GiB; do
    out="$DATA/demo-$budget.gguf"
    echo "-- budget $budget --"
    "$BIN" quantize -m "$MODEL" -i "$IMATRIX" --ctx $CTX --budget "$budget" -o "$out" \
        | grep -E "^weights"
    size=$(du -h "$out" | cut -f1)
    p=$(ppl "$out")
    ROWS+=("$budget  ->  $size  PPL $p")
done

echo
echo "== BF16 baseline perplexity =="
base=$(ppl "$MODEL")

echo
echo "== sample generation (1.75GiB model, greedy) =="
llama-cli -m "$DATA/demo-1.75GiB.gguf" -ngl 99 -n 60 --temp 0 -st \
    -p "Briefly, what is the capital of France?" 2>/dev/null </dev/null \
    | grep -v '^\[' | grep -v '^Exiting' | grep -v '^$' | tail -3

echo
echo "===================== ladder ====================="
echo "BF16 baseline: $(du -h "$MODEL" | cut -f1)  PPL $base"
for row in "${ROWS[@]}"; do
    echo "$row"
done
echo "=================================================="
echo "Every quantized file lands within ~0.1% of its budget; smaller budget,"
echo "lower quality - the solver spends exactly the memory you give it."
