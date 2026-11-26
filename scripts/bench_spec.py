#!/usr/bin/env python3
"""Measure tinyinfer decode tok/s with and without speculative decoding.

    python scripts/bench_spec.py --tinyinfer ../tinyinfer --target tl-25m-f32.gguf \\
        --draft tl-3m-f32.gguf --out results/spec_bench.json

Every configuration runs the tinyinfer CLI on each prompt ``--reps`` times
and records the decode rate it prints (tokens generated / decode wall time,
prefill excluded). The baseline is ``--spec-baseline``: plain decoding through
the same sampling code path as the speculative run. Reported: median over
runs, plus acceptance rate and tokens per target call for speculative runs.
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import time

PROMPTS = ["Once upon a time, there was a little girl named Lily.",
           "Tom and his dog went to the park. They",
           "One day, a big bear found a red ball in the forest."]


def run(cli, args):
    r = subprocess.run([cli] + args, capture_output=True, timeout=600)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace"))
    err = r.stderr.decode(errors="replace")
    m = re.search(r"decode:\s+(\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\)", err)
    out = {"tokens": int(m.group(1)), "ms": float(m.group(2)), "tok_s": float(m.group(3)),
           "text": r.stdout.decode(errors="replace")}
    s = re.search(r"acceptance=([\d.]+)% tokens/round=([\d.]+)", err)
    if s:
        out["acceptance"] = float(s.group(1)) / 100
        out["tokens_per_round"] = float(s.group(2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tinyinfer", required=True, help="tinyinfer checkout (with build/)")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ks", default="1,2,3,4,6,8")
    ap.add_argument("--temps", default="0,0.8")
    ap.add_argument("--paged", action="store_true", help="also time the paged KV cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cli = os.path.join(a.tinyinfer, "build", "tinyinfer.exe" if os.name == "nt" else "tinyinfer")
    common = ["-n", str(a.n), "-t", str(a.threads), "--ignore-eos", "--repeat-penalty", "1",
              "--top-k", "40", "--top-p", "1", "-c", "512"]
    rows = []
    t0 = time.time()
    for temp in [float(x) for x in a.temps.split(",")]:
        configs = [("baseline", ["-m", a.target, "--spec-baseline"]),
                   ("draft alone", ["-m", a.draft, "--spec-baseline"])]
        configs += [(f"spec k={k}", ["-m", a.target, "--draft", a.draft, "--spec-k", str(k)])
                    for k in [int(x) for x in a.ks.split(",")]]
        if a.paged:
            configs.append(("spec k=4 paged", ["-m", a.target, "--draft", a.draft, "--spec-k", "4",
                                               "--kv-paged", "--kv-block", "16"]))
        runs = {name: [] for name, _ in configs}
        # round-robin over configurations so that background load affects all of them alike
        for p_i, prompt in enumerate(PROMPTS):
            for rep in range(a.reps):
                for name, args in configs:
                    runs[name].append(run(cli, args + common + ["--temp", str(temp), "-p", prompt,
                                                                "-s", str(1 + rep + 10 * p_i)]))
        for name, _ in configs:
            rs = runs[name]
            row = {"config": name, "temperature": temp,
                   "tok_s_median": statistics.median(r["tok_s"] for r in rs),
                   "tok_s_all": [r["tok_s"] for r in rs]}
            if "acceptance" in rs[0]:
                row["acceptance_mean"] = statistics.mean(r["acceptance"] for r in rs)
                row["tokens_per_round_mean"] = statistics.mean(r["tokens_per_round"] for r in rs)
            if name == "baseline" and temp == 0:
                row["sample_text"] = rs[0]["text"][:400]
            if name.startswith("spec") and temp == 0:
                row["identical_to_baseline"] = all(
                    x["text"] == y["text"] for x, y in zip(rs, runs["baseline"]))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("tok_s_all", "sample_text")}),
                  flush=True)
    res = {"machine": platform.processor() or platform.machine(), "threads": a.threads,
           "n_predict": a.n, "reps_per_prompt": a.reps, "prompts": PROMPTS,
           "target": os.path.basename(a.target), "draft": os.path.basename(a.draft),
           "date": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
