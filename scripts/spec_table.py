#!/usr/bin/env python3
"""Print the README's speculative-decoding table from results/spec_bench.json."""
import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "results" / "spec_bench.json"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
b = json.load(open(path))
print("| temperature | configuration | decode tok/s, median (min-max) | speed-up | acceptance "
      "| tokens / target call |")
print("|---|---|---:|---:|---:|---:|")
for t in sorted({r["temperature"] for r in b["rows"]}):
    rows = [r for r in b["rows"] if r["temperature"] == t]
    base = next(r for r in rows if r["config"] == "baseline")["tok_s_median"]
    for r in rows:
        acc = f"{r['acceptance_mean'] * 100:.1f} %" if "acceptance_mean" in r else "-"
        tpr = f"{r['tokens_per_round_mean']:.2f}" if "tokens_per_round_mean" in r else "-"
        names = {"baseline": "tl-25m alone", "draft alone": "tl-3m alone (draft)"}
        name = names.get(r["config"], r["config"])
        lo, hi = min(r["tok_s_all"]), max(r["tok_s_all"])
        sp = "1.00×" if r["config"] == "baseline" else (
            "-" if r["config"] == "draft alone" else f"{r['tok_s_median'] / base:.2f}×")
        print(f"| {t:g} | {name} | {r['tok_s_median']:.1f} ({lo:.0f}-{hi:.0f}) | {sp} | {acc} | {tpr} |")
print()
print(f"{b['n_predict']} new tokens per run, {len(b['prompts'])} prompts x {b['reps_per_prompt']} seeds per "
      f"configuration, configurations interleaved round-robin, {b['threads']} threads, top-k 40, "
      f"repetition penalty off, `--ignore-eos`; date {b['date']}.")
