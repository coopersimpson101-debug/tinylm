#!/usr/bin/env python3
"""Regenerate every figure and the banner from the JSON results in results/.

    python docs/assets/make_figures.py

Inputs (all produced by code in this repo, see README "Results"):
  results/train_tl-25m.jsonl, results/train_tl-3m.jsonl   training metrics
  results/spec_bench.json                                 tinyinfer tok/s
Outputs: docs/assets/{hero.png, loss_curve.png, spec_speed.png, banner.svg}
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "docs" / "assets"
ACCENT = "#FB923C"
GREY = "#6e7781"
BLUE = "#3b82f6"


def load_metrics(path):
    train, val = {}, {}
    for line in open(path):
        r = json.loads(line)
        if "val_loss" in r:
            val[r["step"]] = r       # later records (after a resume) win
        elif "loss" in r:
            train[r["step"]] = r
    return [train[k] for k in sorted(train)], [val[k] for k in sorted(val)]


def ema(xs, a=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def loss_axes(ax, runs):
    for name, color in runs:
        tr, va = load_metrics(RES / f"train_{name}.jsonl")
        ax.plot([r["tokens"] / 1e6 for r in tr], ema([r["loss"] for r in tr]), color=color, lw=1.6,
                label=f"{name} train (EMA)")
        ax.plot([r["tokens"] / 1e6 for r in va], [r["val_loss"] for r in va], "o", color=color,
                ms=4, mfc="white", label=f"{name} held-out")
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.set_ylim(1.0, 4.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pretraining on TinyStoriesV2 (RTX 2060, fp16)", fontsize=10)


def speed_axes(ax, bench):
    import numpy as np
    rows = bench["rows"]
    temps = sorted({r["temperature"] for r in rows})
    names = [r["config"] for r in rows if r["temperature"] == temps[0] and r["config"] != "draft alone"]
    w = 0.8 / len(temps)
    for i, t in enumerate(temps):
        vals = {r["config"]: r for r in rows if r["temperature"] == t}
        xs = [j + i * w for j in range(len(names))]
        ys = [vals[n]["tok_s_median"] for n in names]
        q1 = [np.percentile(vals[n]["tok_s_all"], 25) for n in names]
        q3 = [np.percentile(vals[n]["tok_s_all"], 75) for n in names]
        cols = [GREY if n == "baseline" else ACCENT for n in names]
        ax.bar(xs, ys, w * 0.95, color=cols, alpha=1.0 if i == 0 else 0.5,
               yerr=[np.subtract(ys, q1), np.subtract(q3, ys)], capsize=2,
               error_kw={"elinewidth": 0.7, "ecolor": "#444"}, label=f"T={t:g}")
        for x, n, y in zip(xs, names, q3):
            acc = vals[n].get("acceptance_mean")
            if acc is not None:
                ax.text(x, y + 2, f"{acc * 100:.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([j + w * (len(temps) - 1) / 2 for j in range(len(names))])
    ax.set_xticklabels([n.replace("spec ", "") for n in names], fontsize=8)
    base = next(r for r in rows if r["config"] == "baseline" and r["temperature"] == temps[0])
    ax.axhline(base["tok_s_median"], color=GREY, ls="--", lw=0.8)
    draft = next(r for r in rows if r["config"] == "draft alone" and r["temperature"] == temps[0])
    ax.set_ylabel("decode tok/s (median, IQR whiskers)")
    ax.set_title(f"tinyinfer CPU decode: 25.7M target, drafted by the 3.15M model (draft alone "
                 f"{draft['tok_s_median']:.0f} tok/s)\nRyzen 5 3600, {bench['threads']} threads, machine shared "
                 f"with other jobs; labels = acceptance rate", fontsize=8.5)
    from matplotlib.patches import Patch
    handles = [Patch(color=GREY, label="target alone"), Patch(color=ACCENT, label="speculative, T=0 (greedy)"),
               Patch(color=ACCENT, alpha=0.5, label="speculative, T=0.8")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def banner(runs):
    tr, _ = load_metrics(RES / f"train_{runs[0][0]}.jsonl")
    xs = [r["tokens"] for r in tr]
    ys = ema([r["loss"] for r in tr], 0.95)
    x0, x1 = 700, 1230
    y0, y1 = 70, 330
    lo, hi = 1.0, max(ys[:3] + [4.0])
    pts = " ".join(f"{x0 + (x - xs[0]) / (xs[-1] - xs[0]) * (x1 - x0):.1f},"
                   f"{y1 - (min(y, hi) - lo) / (hi - lo) * (y1 - y0):.1f}" for x, y in zip(xs, ys))
    grid = "".join(f'<line x1="{x}" y1="0" x2="{x}" y2="400" stroke="#161b22" stroke-width="1"/>'
                   for x in range(0, 1281, 40))
    grid += "".join(f'<line x1="0" y1="{y}" x2="1280" y2="{y}" stroke="#161b22" stroke-width="1"/>'
                    for y in range(0, 401, 40))
    final = ys[-1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400">
<rect width="1280" height="400" fill="#0d1117"/>
{grid}
<text x="64" y="190" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="84" font-weight="bold" fill="#e6edf3">tinylm</text>
<text x="66" y="240" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="20" fill="#8b949e">a Qwen2-style LLM trained from scratch at home,</text>
<text x="66" y="274" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="20" fill="#8b949e">served by tinyinfer with speculative decoding</text>
<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#30363d" stroke-width="1.5"/>
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#30363d" stroke-width="1.5"/>
<polyline points="{pts}" fill="none" stroke="{ACCENT}" stroke-width="3" stroke-linejoin="round"/>
<text x="{x1}" y="{y1 + 28}" text-anchor="end" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="16" fill="#8b949e">real training loss, {runs[0][0]}: {ys[0]:.1f} -&gt; {final:.2f} nats</text>
</svg>
'''
    (OUT / "banner.svg").write_text(svg, encoding="utf-8")


def main():
    runs = [("tl-25m", ACCENT), ("tl-3m", BLUE)]
    runs = [r for r in runs if (RES / f"train_{r[0]}.jsonl").exists()]
    bench = json.load(open(RES / "spec_bench.json")) if (RES / "spec_bench.json").exists() else None
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    loss_axes(ax, runs)
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png")
    if bench:
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
        speed_axes(ax, bench)
        fig.tight_layout()
        fig.savefig(OUT / "spec_speed.png")
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130, gridspec_kw={"width_ratios": [1, 1.15]})
        loss_axes(axs[0], runs)
        speed_axes(axs[1], bench)
        fig.tight_layout()
        fig.savefig(OUT / "hero.png")
    banner(runs)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
