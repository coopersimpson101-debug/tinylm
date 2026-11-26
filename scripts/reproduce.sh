#!/usr/bin/env bash
# Reproduce every result in the README: data -> tokenizer -> two models -> eval -> GGUF -> tinyinfer bench.
# Usage: scripts/reproduce.sh [DATA_DIR] [RUN_DIR] [TINYINFER_DIR]
# These are the commands behind the committed results (see results/*.json). The committed tl-25m run
# was interrupted and resumed on a shared machine (README "Notes on this run"); this runs it in one go.
set -euo pipefail
DATA=${1:-data}
RUNS=${2:-runs}
TI=${3:-../tinyinfer}

# 1. data (1 GB prefix of TinyStoriesV2-GPT4-train + the full valid split, CDLA-Sharing-1.0)
python -m tinylm.download tinystories --out "$DATA/raw" --max-mb 1000
python -m tinylm.prepare --train "$DATA/raw/tinystories-train.txt" --train-truncated \
    --valid "$DATA/raw/tinystories-valid.txt" --out "$DATA/ts4k" --vocab 4096 --tok-mb 100 --workers 8

# 2. target model (25.7M params) and draft model (3.15M params)
python -m tinylm.train --data "$DATA/ts4k" --out "$RUNS/tl-25m" --model tl-25m --seq-len 512 \
    --batch-size 16 --grad-accum 2 --max-steps 2250 --warmup 200 --lr 1e-3 --min-lr 1e-4 \
    --eval-every 250 --eval-batches 40 --ckpt-every 250
python -m tinylm.train --data "$DATA/ts4k" --out "$RUNS/tl-3m" --model tl-3m --seq-len 512 \
    --batch-size 16 --grad-accum 2 --max-steps 1500 --warmup 100 --lr 2e-3 --min-lr 2e-4 \
    --eval-every 250 --eval-batches 40 --ckpt-every 250

# 3. evaluation
for m in tl-25m tl-3m; do
    python -m tinylm.evaluate "$RUNS/$m/model.pt" --data "$DATA/ts4k" --out "results/eval_$m.json"
    cp "$RUNS/$m/metrics.jsonl" "results/train_$m.jsonl"
done

# 4. export to GGUF and benchmark speculative decoding in tinyinfer
for m in tl-25m tl-3m; do
    python -m tinylm.export "$RUNS/$m/model.pt" "$DATA/ts4k/tokenizer.json" -o "$RUNS/$m.gguf" --name "$m"
done
python scripts/bench_spec.py --tinyinfer "$TI" --target "$RUNS/tl-25m.gguf" --draft "$RUNS/tl-3m.gguf" \
    --threads 3 --reps 3 --paged --out results/spec_bench.json
python docs/assets/make_figures.py
