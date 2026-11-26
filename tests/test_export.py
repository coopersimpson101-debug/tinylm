"""GGUF export, and logits parity with tinyinfer's NumPy reference and C engine."""
import sys

import numpy as np
import pytest
import torch
from conftest import tinyinfer_dir

from tinylm import gguf, tasks
from tinylm.config import get_config
from tinylm.export import export
from tinylm.model import Transformer, token_logprobs

TEXT = "Once upon a time, there was a kid named Tom. Tom had a happy ball."


@pytest.fixture(scope="module")
def exported(tiny_data, tmp_path_factory):
    d, tok = tiny_data
    torch.manual_seed(0)
    m = Transformer(get_config("smoke", vocab_size=tok.vocab_size, rope_theta=10000.0)).eval()
    with torch.no_grad():  # non-trivial norms/biases so every tensor matters
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.add_(0.1 * torch.randn_like(p))
    out = tmp_path_factory.mktemp("gguf")
    f32 = str(export(m, tok, str(out / "m-f32.gguf")))
    f16 = str(export(m, tok, str(out / "m-f16.gguf"), "f16"))
    return m, tok, f32, f16


def test_gguf_roundtrip(exported):
    m, tok, f32, f16 = exported
    kv, t = gguf.read(f32)
    assert kv["general.architecture"] == "qwen2"
    assert kv["qwen2.attention.head_count_kv"] == m.cfg.n_kv_heads
    assert kv["tokenizer.ggml.tokens"] == tok.tokens
    assert kv["tokenizer.ggml.merges"] == tok.merge_strings
    assert np.array_equal(t["blk.1.ffn_down.weight"], m.layers[1].mlp.down.weight.detach().numpy())
    assert "output.weight" not in t  # tied embeddings
    _, t16 = gguf.read(f16)
    assert np.allclose(t16["token_embd.weight"], t["token_embd.weight"], atol=1e-3)
    assert t16["blk.0.attn_q.bias"].dtype == np.float32


def _tinyinfer():
    d = tinyinfer_dir()
    if d is None:
        pytest.skip("tinyinfer checkout not found (set TINYINFER_DIR)")
    sys.path.insert(0, str(d / "python"))
    return d


def test_tinyinfer_python_reference_matches_torch(exported):
    _tinyinfer()
    from tinyinfer_ref.model import QwenRef
    m, tok, f32, _ = exported
    ids = [tok.eos_id] + tok.encode(TEXT)
    ref = QwenRef(f32).forward(ids)
    with torch.no_grad():
        ours = m(torch.tensor([ids]))[0].double().numpy()
    assert np.abs(ref - ours).max() < 2e-4


def test_tinyinfer_c_engine_matches_torch(exported):
    d = _tinyinfer()
    import os
    os.environ.setdefault("TI_BUILD_DIR", str(d / "build"))
    from tinyinfer_ref import capi
    lib_path = capi.BUILD / capi.LIBNAME
    if not lib_path.exists():
        pytest.skip(f"tinyinfer library not built ({lib_path}); run python build.py there")
    lib = capi.load_lib()
    m, tok, f32, f16 = exported
    ids = [tok.eos_id] + tok.encode(TEXT)
    with torch.no_grad():
        ours = m(torch.tensor([ids]))[0].numpy()
    for path, tol in ((f32, 1e-3), (f16, 0.05)):
        cm = capi.Model(lib, path, ctx=128, threads=2)
        try:
            got = cm.forward(ids)
            assert np.abs(got - ours).max() < tol, (path, np.abs(got - ours).max())
            assert (got.argmax(-1) == ours.argmax(-1)).mean() > 0.95
            # tokenizer parity: tinyinfer's C BPE produces our ids
            assert cm.tokenize(TEXT, parse_special=False) == tok.encode(TEXT)
            assert cm.detokenize_bytes(ids[1:]) == TEXT.encode()
        finally:
            cm.close()


def test_continuation_logprob_consistent(exported):
    m, tok, _, _ = exported
    ctx, end = "Tom had a ball.", "Lily was happy."
    lp = tasks.continuation_logprob(m, tok, ctx, [end], "cpu", 128)[0]
    ids = [tok.eos_id] + tok.encode(ctx) + tok.encode(" " + end)
    with torch.no_grad():
        full = token_logprobs(m, torch.tensor([ids]))[0]
    n_end = len(tok.encode(" " + end))
    assert abs(lp - float(full[-n_end:].sum())) < 1e-4


def test_task_construction(stories):
    pairs = tasks.grammar_pairs()
    assert len(pairs) > 150 and all(g != b for _, g, b in pairs)
    docs = [s + f" Then they slept. The sun set. It was day {i}." for i, s in enumerate(stories)]
    items = tasks.cloze_items(docs, 50)
    assert len(items) == 50
    for ctx, ch, gold in items:
        assert len(set(ch)) == 4 and 0 <= gold < 4 and ctx
