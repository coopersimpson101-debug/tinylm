"""Pretraining loop.

    python -m tinylm.train --data data/ts4k --out runs/tl-25m --model tl-25m \\
        --batch-size 32 --grad-accum 2 --max-steps 6000

Features: hand-written AdamW + cosine/warm-up schedule + global-norm clipping
(``tinylm.optim``), gradient accumulation, fp16 autocast with dynamic loss
scaling on CUDA (bf16 where the GPU supports it; fp32 on CPU), periodic
held-out evaluation, atomic checkpoints with exact resume (the data sampler is
a pure function of the step), JSONL metrics, and an optional wall-clock limit.

Data parallelism: ``train_step`` is a pure function of (model, optimiser,
micro-batches, lr) with a ``grad_sync`` hook that runs between backward and the
optimiser step. Under ``torchrun`` (or any launcher that sets RANK /
WORLD_SIZE / MASTER_ADDR) ``grad_sync`` all-reduces gradients with
``torch.distributed`` and every rank samples an independent data stream; an
external driver (e.g. a ring all-reduce implementation) can pass its own hook.
"""
import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from . import optim
from .config import ModelConfig, get_config
from .model import Transformer
from .shards import TokenDataset


@dataclass
class TrainConfig:
    data: str = "data/ts4k"
    out: str = "runs/run"
    model: str = "tl-25m"
    seq_len: int = 512
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 5000
    warmup: int = 200
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int = 500
    log_every: int = 10
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1337
    time_limit_min: float = 0.0
    stop_at_step: int = 0  # leave the run early at this step (checkpointed), e.g. to split a job
    threads: int = 0
    model_overrides: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- distributed
def dist_info():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def init_distributed(device: str):
    rank, world = dist_info()
    if world > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.startswith("cuda") else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return rank, world


def allreduce_grad_sync(model):
    """Average gradients over all ranks (the job DDP would do, without bucketing)."""
    world = torch.distributed.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world)


# ----------------------------------------------------------------------------- step
def train_step(model, opt, scaler, micro_batches, lr: float, clip: float,
               amp_ctx=contextlib.nullcontext, grad_sync=None) -> dict:
    """One optimiser step over ``len(micro_batches)`` accumulated micro-batches."""
    model.train()
    opt.zero_grad()
    n = len(micro_batches)
    total = 0.0
    for x, y in micro_batches:
        with amp_ctx():
            _, loss = model(x, y)
        scaler.scale_loss(loss / n).backward()
        total += float(loss.detach())
    if grad_sync is not None:
        grad_sync(model)
    params = list(opt.params())
    finite = scaler.unscale_and_check(params)
    gnorm = optim.clip_grad_norm(params, clip) if finite else float("inf")
    if finite:
        opt.step(lr)
    scaler.update(finite)
    return {"loss": total / n, "grad_norm": gnorm, "skipped": not finite}


@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batch_size: int, max_batches: int, device,
             amp_ctx=contextlib.nullcontext) -> float:
    """Mean token NLL over the first ``max_batches`` non-overlapping validation windows."""
    model.eval()
    tot, cnt = 0.0, 0
    for x, y in dataset.sequential(batch_size, max_batches):
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        with amp_ctx():
            _, loss = model(x, y)
        tot += float(loss) * y.numel()
        cnt += y.numel()
    model.train()
    return tot / max(cnt, 1)


# ----------------------------------------------------------------------------- setup helpers
def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(name: str, device: str) -> str:
    if name != "auto":
        return name
    if device.startswith("cuda"):
        return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    return "fp32"


def amp_context(device: str, dtype: str):
    if dtype == "fp32":
        return contextlib.nullcontext
    dt = torch.float16 if dtype == "fp16" else torch.bfloat16
    dev = "cuda" if device.startswith("cuda") else "cpu"
    return lambda: torch.autocast(device_type=dev, dtype=dt)


def save_checkpoint(path, model, opt, scaler, step, tokens, tc: TrainConfig, mc: ModelConfig):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "tokens": tokens, "train_config": asdict(tc),
                "model_config": mc.to_dict()}, tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu") -> Transformer:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Transformer(ModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


# ----------------------------------------------------------------------------- main loop
def run(tc: TrainConfig) -> dict:
    device = pick_device(tc.device)
    rank, world = init_distributed(device)
    if world > 1 and device == "cuda":
        device = f"cuda:{rank % torch.cuda.device_count()}"
    dtype = pick_dtype(tc.dtype, device)
    if tc.threads:
        torch.set_num_threads(tc.threads)
    torch.manual_seed(tc.seed)
    master = rank == 0
    os.makedirs(tc.out, exist_ok=True)

    mc = get_config(tc.model, max_seq_len=max(tc.seq_len, PRESET_CTX(tc.model)), **tc.model_overrides)
    train_ds = TokenDataset(os.path.join(tc.data, "train_*.bin"), tc.seq_len, seed=tc.seed,
                            rank=rank, world=world)
    valid_ds = TokenDataset(os.path.join(tc.data, "valid_*.bin"), tc.seq_len)
    model = Transformer(mc).to(device)
    opt = optim.AdamW(optim.param_groups(model, tc.weight_decay), lr=tc.lr,
                      betas=(tc.beta1, tc.beta2))
    scaler = optim.LossScaler(enabled=(dtype == "fp16"))
    amp = amp_context(device, dtype)
    grad_sync = allreduce_grad_sync if world > 1 else None

    ckpt = os.path.join(tc.out, "ckpt.pt")
    step, tokens = 0, 0
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, tokens = ck["step"], ck["tokens"]
        if master:
            print(f"resumed from step {step} ({tokens} tokens)", flush=True)
    log_path = os.path.join(tc.out, "metrics.jsonl")
    if master:
        json.dump({"train": asdict(tc), "model": mc.to_dict(), "device": device, "dtype": dtype,
                   "world": world, "n_params": mc.n_params(),
                   "n_params_non_embedding": mc.n_params(False),
                   "train_tokens_available": train_ds.n_tokens,
                   "torch": torch.__version__,
                   "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda")
                   else "cpu"}, open(os.path.join(tc.out, "config.json"), "w"), indent=2)
        print(f"model {tc.model}: {mc.n_params() / 1e6:.2f}M params on {device} ({dtype}), "
              f"world={world}, train tokens available {train_ds.n_tokens / 1e6:.0f}M", flush=True)
    logf = open(log_path, "a") if master else None
    if master and step > 0:
        logf.write(json.dumps({"resume_from_step": step, "max_steps": tc.max_steps}) + "\n")
    tok_per_step = tc.batch_size * tc.grad_accum * tc.seq_len * world
    t_start = time.time()
    t_last = time.time()
    last = {}
    while step < tc.max_steps:
        lr = optim.cosine_lr(step, tc.lr, tc.min_lr, tc.warmup, tc.max_steps)
        micro = []
        for k in range(tc.grad_accum):
            x, y = train_ds.batch(step * tc.grad_accum + k, tc.batch_size)
            micro.append((torch.from_numpy(x).to(device, non_blocking=True),
                          torch.from_numpy(y).to(device, non_blocking=True)))
        last = train_step(model, opt, scaler, micro, lr, tc.clip, amp, grad_sync)
        step += 1
        tokens += tok_per_step
        if master and (step % tc.log_every == 0 or step == 1):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t_last
            t_last = time.time()
            n_steps = tc.log_every if step > 1 else 1
            rec = {"step": step, "tokens": tokens, "loss": round(last["loss"], 5), "lr": lr,
                   "grad_norm": round(last["grad_norm"], 4), "skipped": last["skipped"],
                   "tok_s": round(tok_per_step * n_steps / dt, 1),
                   "elapsed_s": round(time.time() - t_start, 1)}
            if scaler.enabled:
                rec["loss_scale"] = scaler.scale
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:6d} | loss {last['loss']:.4f} | lr {lr:.2e} | gnorm "
                  f"{last['grad_norm']:.3f} | {rec['tok_s']:.0f} tok/s", flush=True)
        out_of_time = (tc.time_limit_min > 0 and time.time() - t_start > tc.time_limit_min * 60) \
            or step == tc.stop_at_step
        final = step == tc.max_steps or out_of_time
        if master and (step % tc.eval_every == 0 or final):
            vl = evaluate(model, valid_ds, tc.batch_size, tc.eval_batches, device, amp)
            rec = {"step": step, "tokens": tokens, "val_loss": round(vl, 5),
                   "val_ppl": round(math.exp(vl), 4), "elapsed_s": round(time.time() - t_start, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:6d} | val loss {vl:.4f} | val ppl {math.exp(vl):.3f}", flush=True)
        if master and (step % tc.ckpt_every == 0 or final):
            save_checkpoint(ckpt, model, opt, scaler, step, tokens, tc, mc)
        if out_of_time:
            if master:
                print(f"stopping early at step {step} (time limit / stop_at_step)", flush=True)
            break
    if master:
        torch.save({"model": model.state_dict(), "model_config": mc.to_dict(), "step": step,
                    "tokens": tokens}, os.path.join(tc.out, "model.pt"))
        logf.close()
    return {"step": step, "tokens": tokens, **last}


def PRESET_CTX(name):  # noqa: N802
    from .config import PRESETS
    return PRESETS[name].max_seq_len


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in TrainConfig.__dataclass_fields__.values():
        if f.name == "model_overrides":
            continue
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--override", action="append", default=[],
                    help="model config override, e.g. --override n_layers=4")
    a = ap.parse_args(argv)
    ov = {}
    for o in a.override:
        k, v = o.split("=", 1)
        ov[k] = type(getattr(ModelConfig(), k))(v)
    kw = {k: v for k, v in vars(a).items() if k != "override"}
    run(TrainConfig(**kw, model_overrides=ov))


if __name__ == "__main__":
    main()
