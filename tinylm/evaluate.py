"""Evaluation harness: held-out perplexity, zero-shot tasks and samples.

    python -m tinylm.evaluate runs/tl-25m/model.pt --data data/ts4k --out results/tl-25m.json

* perplexity: exp(mean NLL) over non-overlapping ``seq_len`` windows of the
  validation shards (TinyStoriesV2 *valid* split, never trained on);
* ``story_cloze`` and ``grammar`` (see ``tinylm.tasks``);
* a few sampled continuations of fixed prompts with a fixed seed.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from . import tasks
from .bpe import Tokenizer
from .shards import TokenDataset, read_shard
from .train import load_model

PROMPTS = ["Once upon a time,", "Lily wanted to play outside, but", "Tom found a small box. Inside"]


@torch.no_grad()
def perplexity(model, pattern, seq_len, batch_size, device, max_batches=None):
    ds = TokenDataset(pattern, seq_len)
    tot = cnt = 0.0
    for x, y in ds.sequential(batch_size, max_batches):
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        logits = model(x).float()
        nll = torch.logsumexp(logits, -1) - logits.gather(2, y[..., None]).squeeze(-1)
        tot += float(nll.sum())
        cnt += y.numel()
    loss = tot / cnt
    return {"loss": loss, "ppl": math.exp(loss), "tokens": int(cnt)}


def decode_stories(pattern_file, tok, max_stories=5000):
    toks = np.asarray(read_shard(pattern_file)[: max_stories * 400], dtype=np.int64)
    cuts = np.flatnonzero(toks == tok.eos_id)
    out, prev = [], 0
    for c in cuts[:max_stories]:
        out.append(tok.decode(toks[prev:c].tolist()))
        prev = c + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--ppl-batches", type=int, default=0, help="0 = whole validation split")
    ap.add_argument("--cloze-items", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    model = load_model(a.checkpoint, a.device)
    tok = Tokenizer.load(os.path.join(a.data, "tokenizer.json"))
    res = {"checkpoint": os.path.basename(os.path.dirname(os.path.abspath(a.checkpoint))),
           "n_params": model.cfg.n_params(), "device": a.device}
    res["perplexity"] = perplexity(model, os.path.join(a.data, "valid_*.bin"), a.seq_len, 16,
                                   a.device, a.ppl_batches or None)
    stories = decode_stories(os.path.join(a.data, "valid_000.bin"), tok)
    res["story_cloze"] = tasks.run_cloze(model, tok, stories, a.device, a.cloze_items, a.seq_len)
    res["grammar"] = tasks.run_grammar(model, tok, a.device, a.seq_len)
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    res["samples"] = []
    for p in PROMPTS:
        for _ in range(a.samples):
            ids = torch.tensor([[tok.eos_id] + tok.encode(p)], device=a.device)
            out = model.generate(ids, 160, temperature=0.8, top_k=40, eos_id=tok.eos_id, generator=g)
            text = tok.decode([t for t in out[0, 1:].tolist() if t != tok.eos_id])
            res["samples"].append({"prompt": p, "text": text})
    res["seconds"] = round(time.time() - t0, 1)
    js = json.dumps(res, indent=2)
    print(js)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(js + "\n")


if __name__ == "__main__":
    main()
