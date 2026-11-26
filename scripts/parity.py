#!/usr/bin/env python3
"""Compare logits of a trained checkpoint in PyTorch vs its GGUF export in tinyinfer.

    python scripts/parity.py runs/tl-25m/model.pt runs/tl-25m-f32.gguf data/ts4k/tokenizer.json \\
        --tinyinfer ../tinyinfer --out results/parity_tl-25m.json

Runs tinyinfer's NumPy reference (float64) and its C engine (via ctypes) on the
same token ids and reports max |diff| of the logits, greedy-argmax agreement,
and whether tinyinfer's C tokenizer reproduces tinylm's token ids.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinylm.bpe import Tokenizer  # noqa: E402
from tinylm.train import load_model  # noqa: E402

TEXT = ("Once upon a time, there was a little dog named Max. Max liked to run in the park. "
        "One day, he saw a big red ball under a tree. \"Can I play with you?\" asked a bird.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("gguf")
    ap.add_argument("tokenizer")
    ap.add_argument("--tinyinfer", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.tinyinfer) / "python"))
    from tinyinfer_ref import capi
    from tinyinfer_ref.model import QwenRef
    tok = Tokenizer.load(a.tokenizer)
    ids = [tok.eos_id] + tok.encode(TEXT)
    with torch.no_grad():
        ours = load_model(a.checkpoint)(torch.tensor([ids]))[0].double().numpy()
    ref = QwenRef(a.gguf).forward(ids)
    lib = capi.load_lib()
    m = capi.Model(lib, a.gguf, ctx=512, threads=4)
    c = m.forward(ids).astype(np.float64)
    c_tok = m.tokenize(TEXT, parse_special=False)
    m.close()
    res = {"n_tokens": len(ids), "logit_std": float(ours.std()),
           "max_abs_diff_numpy_ref": float(np.abs(ref - ours).max()),
           "max_abs_diff_c_engine": float(np.abs(c - ours).max()),
           "argmax_agreement_c_engine": float((c.argmax(-1) == ours.argmax(-1)).mean()),
           "c_tokenizer_matches": c_tok == ids[1:]}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
