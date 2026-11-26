"""Model maths, optimiser and the training loop."""
import json
import math
import os

import numpy as np
import pytest
import torch

from tinylm import optim
from tinylm.config import PRESETS, get_config
from tinylm.model import RMSNorm, Transformer, apply_rope, rope_tables
from tinylm.train import TrainConfig, evaluate, run, train_step

torch.set_num_threads(2)


def small(**kw):
    torch.manual_seed(0)
    return Transformer(get_config("smoke", **kw))


@pytest.mark.parametrize("name", list(PRESETS))
def test_param_count_formula(name):
    c = PRESETS[name]
    if c.n_params() > 40e6:  # avoid allocating the big ones in CI
        return
    m = Transformer(c)
    assert sum(p.numel() for p in m.parameters()) == c.n_params()


def test_rmsnorm_formula():
    n = RMSNorm(8, 1e-6)
    with torch.no_grad():
        n.weight.copy_(torch.arange(1.0, 9.0))
    x = torch.randn(3, 8)
    ref = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * n.weight
    assert torch.allclose(n(x), ref, atol=1e-6)


def test_rope_is_relative():
    cos, sin = rope_tables(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return float((qi * kj).sum())
    assert abs(score(5, 2) - score(40, 37)) < 1e-4
    assert abs(score(0, 0) - float((q * k).sum())) < 1e-5


def test_causal_and_kv_cache_consistent():
    m = small().eval()
    x = torch.randint(0, 512, (2, 20))
    full = m(x)
    x2 = x.clone()
    x2[:, 15:] = 7
    assert torch.allclose(m(x2)[:, :15], full[:, :15], atol=1e-5)  # no peeking ahead
    caches = [dict() for _ in m.layers]
    steps = [m(x[:, :6], caches=caches)]
    for t in range(6, 20):
        steps.append(m(x[:, t:t + 1], start_pos=t, caches=caches))
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-4)


def test_gqa_equals_mha_with_repeated_kv():
    g = small()                      # 4 heads, 2 kv heads
    torch.manual_seed(1)
    h = Transformer(get_config("smoke", n_kv_heads=4))
    sd = g.state_dict()
    hd = g.cfg.head_dim
    new = {}
    for k, v in sd.items():
        if any(s in k for s in ("wk.", "wv.")):
            parts = v.view(2, hd, *v.shape[1:])
            v = parts.repeat_interleave(2, dim=0).reshape(4 * hd, *v.shape[1:])
        new[k] = v
    h.load_state_dict(new)
    x = torch.randint(0, 512, (1, 12))
    assert torch.allclose(g(x), h(x), atol=1e-5)


def test_loss_is_log_vocab_at_init():
    m = small()
    x = torch.randint(0, 512, (4, 32))
    _, loss = m(x, torch.randint(0, 512, (4, 32)))
    assert abs(float(loss) - math.log(512)) < 0.1


def test_adamw_matches_torch():
    torch.manual_seed(0)
    a = small()
    b = small()
    b.load_state_dict(a.state_dict())
    oa = optim.AdamW(optim.param_groups(a, 0.1), lr=3e-3, betas=(0.9, 0.95))
    gb = optim.param_groups(b, 0.1)
    ob = torch.optim.AdamW(gb, lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
    for s in range(15):
        x = torch.randint(0, 512, (2, 16), generator=torch.Generator().manual_seed(s))
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            m(x, x)[1].backward()
            o.step()
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), n


def test_schedule_and_clipping():
    lrs = [optim.cosine_lr(s, 1.0, 0.1, 10, 110) for s in range(120)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.55) and lrs[110] == pytest.approx(0.1)
    assert all(x >= y for x, y in zip(lrs[9:110], lrs[10:111]))
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    assert optim.clip_grad_norm([p], 1.0) == pytest.approx(5.0)
    assert float(p.grad.norm()) == pytest.approx(1.0, rel=1e-5)


def test_loss_scaler_skips_non_finite_steps():
    s = optim.LossScaler(True, init_scale=1024.0, growth_interval=2)
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([float("inf"), 1.0])
    assert not s.unscale_and_check([p])
    s.update(False)
    assert s.scale == 512.0
    p.grad = torch.tensor([512.0, 1024.0])
    assert s.unscale_and_check([p]) and torch.equal(p.grad, torch.tensor([1.0, 2.0]))
    s.update(True)
    s.update(True)
    assert s.scale == 1024.0


def test_grad_accumulation_equals_big_batch():
    a, b = small(), small()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 512, (4, 16), generator=torch.Generator().manual_seed(3))
    oa = optim.AdamW(optim.param_groups(a, 0.0), lr=1e-3)
    ob = optim.AdamW(optim.param_groups(b, 0.0), lr=1e-3)
    off = optim.LossScaler(False)
    train_step(a, oa, off, [(x, x)], 1e-3, 1.0)
    train_step(b, ob, off, [(x[:2], x[:2]), (x[2:], x[2:])], 1e-3, 1.0)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def _cfg(d, out, **kw):
    base = dict(data=str(d), out=str(out), model="smoke", seq_len=32, batch_size=8, max_steps=40,
                warmup=5, lr=3e-3, min_lr=3e-4, eval_every=20, eval_batches=2, ckpt_every=20,
                log_every=10, device="cpu", dtype="fp32", threads=2)
    base.update(kw)
    return TrainConfig(**base)


def test_training_reduces_loss_and_resume_is_exact(tiny_data, tmp_path):
    d, tok = tiny_data
    ov = {"vocab_size": tok.vocab_size}
    r1 = run(_cfg(d, tmp_path / "a", model_overrides=ov))
    lines = [json.loads(line) for line in open(tmp_path / "a" / "metrics.jsonl")]
    vals = [x["val_loss"] for x in lines if "val_loss" in x]
    assert vals[-1] < math.log(tok.vocab_size) - 1.0
    # stop at step 20, resume to 40: must equal the uninterrupted run
    run(_cfg(d, tmp_path / "b", stop_at_step=20, model_overrides=ov))
    ck = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    assert ck["step"] == 20
    r2 = run(_cfg(d, tmp_path / "b", max_steps=40, model_overrides=ov))
    assert r2["step"] == 40 and r1["loss"] == pytest.approx(r2["loss"], abs=0, rel=0)
    m1 = torch.load(tmp_path / "a" / "model.pt", weights_only=False)["model"]
    m2 = torch.load(tmp_path / "b" / "model.pt", weights_only=False)["model"]
    assert all(torch.equal(m1[k], m2[k]) for k in m1)


def test_evaluate_and_generate(tiny_data):
    from tinylm.shards import TokenDataset
    d, tok = tiny_data
    m = small(vocab_size=tok.vocab_size)
    ds = TokenDataset(os.path.join(d, "valid_*.bin"), 32)
    vl = evaluate(m, ds, 4, 2, "cpu")
    assert abs(vl - math.log(tok.vocab_size)) < 0.3
    out = m.generate(torch.tensor([[tok.eos_id]]), 10, temperature=0.0)
    assert out.shape == (1, 11)
    g1 = m.generate(torch.tensor([[1, 2]]), 8, temperature=1.0, generator=torch.Generator().manual_seed(0))
    g2 = m.generate(torch.tensor([[1, 2]]), 8, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.equal(g1, g2)


def _ddp_worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world))
    import torch.distributed as dist
    from tinylm.train import allreduce_grad_sync
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(0)
    m = Transformer(get_config("smoke"))
    o = optim.AdamW(optim.param_groups(m, 0.1), lr=1e-3)
    x = torch.randint(0, 512, (4, 16), generator=torch.Generator().manual_seed(9))
    shard = x[rank * 2:(rank + 1) * 2]
    train_step(m, o, optim.LossScaler(False), [(shard, shard)], 1e-3, 1.0, grad_sync=allreduce_grad_sync)
    if rank == 0:
        torch.save(m.state_dict(), out)
    dist.destroy_process_group()


def test_ddp_grad_sync_equals_single_process(tmp_path):
    import socket
    import torch.multiprocessing as mp
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    s = socket.socket()
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    out = str(tmp_path / "ddp.pt")
    mp.spawn(_ddp_worker, args=(2, port, out), nprocs=2, join=True)
    torch.manual_seed(0)
    m = Transformer(get_config("smoke"))
    o = optim.AdamW(optim.param_groups(m, 0.1), lr=1e-3)
    x = torch.randint(0, 512, (4, 16), generator=torch.Generator().manual_seed(9))
    # world=2 with 2 rows each == accumulation over the same two halves
    train_step(m, o, optim.LossScaler(False), [(x[:2], x[:2]), (x[2:], x[2:])], 1e-3, 1.0)
    got = torch.load(out)
    for k, v in m.state_dict().items():
        assert torch.allclose(v, got[k], atol=1e-6), k
    assert np.isfinite(float(next(iter(got.values())).sum()))
