"""Optimiser pieces written by hand: AdamW, LR schedule, gradient clipping, loss scaling.

AdamW (Kingma & Ba 2015; Loshchilov & Hutter 2019, decoupled weight decay)::

    theta <- theta (1 - lr * lambda)
    m <- b1 m + (1 - b1) g            v <- b2 v + (1 - b2) g^2
    theta <- theta - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)

(the same operation order as ``torch.optim.AdamW``, so the two agree to
rounding; ``tests/test_optim.py`` checks this). Weight decay is applied only
to matrices (not to norms, biases or -- by convention -- embeddings).

Cosine schedule with linear warm-up::

    lr(s) = lr_max * (s + 1) / W                                   s < W
    lr(s) = lr_min + (lr_max - lr_min) * (1 + cos(pi * (s - W) / (S - W))) / 2

Global-norm clipping: g <- g * min(1, c / ||g||_2) over all parameters.

Dynamic loss scaling for fp16 (Micikevicius et al. 2018): multiply the loss by
S before backward, divide gradients by S, skip the step and halve S if any
gradient is non-finite, double S after ``growth_interval`` clean steps.
"""
import math

import torch


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "tok_embd" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class AdamW:
    def __init__(self, groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8):
        self.groups = [dict(g) for g in groups]
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.t = 0
        self.state = {}

    def params(self):
        for g in self.groups:
            yield from g["params"]

    def zero_grad(self):
        for p in self.params():
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float | None = None):
        lr = self.lr if lr is None else lr
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1 = 1 - b1 ** self.t
        bc2_sqrt = math.sqrt(1 - b2 ** self.t)
        for g in self.groups:
            wd = g["weight_decay"]
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            for p in ps:
                if p not in self.state:
                    self.state[p] = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            gs = [p.grad for p in ps]
            ms = [self.state[p]["m"] for p in ps]
            vs = [self.state[p]["v"] for p in ps]
            # multi-tensor ("foreach") primitives: same arithmetic as the per-tensor loop
            #   p *= 1 - lr wd;  m = lerp(m, g, 1-b1);  v = b2 v + (1-b2) g^2
            #   p -= lr/bc1 * m / (sqrt(v)/sqrt(bc2) + eps)
            # but one kernel launch per op for the whole group (matters when the CPU is busy).
            if wd:
                torch._foreach_mul_(ps, 1 - lr * wd)
            torch._foreach_lerp_(ms, gs, 1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, gs, gs, value=1 - b2)
            denom = torch._foreach_sqrt(vs)
            torch._foreach_div_(denom, bc2_sqrt)
            torch._foreach_add_(denom, self.eps)
            torch._foreach_addcdiv_(ps, ms, denom, value=-lr / bc1)

    def state_dict(self):
        index = {id(p): i for i, p in enumerate(self.params())}
        return {"t": self.t, "state": {index[id(p)]: s for p, s in self.state.items()}}

    def load_state_dict(self, d):
        ps = list(self.params())
        self.t = d["t"]
        self.state = {ps[int(i)]: {k: v.to(ps[int(i)].device) for k, v in s.items()}
                      for i, s in d["state"].items()}


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def global_grad_norm(params) -> torch.Tensor:
    gs = [p.grad for p in params if p.grad is not None]
    if not gs:
        return torch.tensor(0.0)
    norms = torch._foreach_norm(gs)  # one L2 norm per tensor, then the norm of norms
    return torch.stack([n.float() for n in norms]).norm()


@torch.no_grad()
def clip_grad_norm(params, max_norm: float) -> float:
    params = [p for p in params if p.grad is not None]
    norm = global_grad_norm(params)
    if max_norm > 0:
        scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
        torch._foreach_mul_([p.grad for p in params], scale)
    return float(norm)


class LossScaler:
    """Dynamic loss scaling; ``enabled=False`` makes every method a no-op."""

    def __init__(self, enabled: bool, init_scale: float = 2.0 ** 15, growth_interval: int = 1000):
        self.enabled = enabled
        self.scale = init_scale if enabled else 1.0
        self.growth_interval = growth_interval
        self.good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale if self.enabled else loss

    @torch.no_grad()
    def unscale_and_check(self, params) -> bool:
        """Divide grads by the scale; return True when all gradients are finite."""
        if not self.enabled:
            return True
        gs = [p.grad for p in params if p.grad is not None]
        torch._foreach_mul_(gs, 1.0 / self.scale)
        norms = torch._foreach_norm(gs)  # inf/nan anywhere -> non-finite norm
        return bool(torch.isfinite(torch.stack(norms)).all())

    def update(self, finite: bool):
        if not self.enabled:
            return
        if not finite:
            self.scale /= 2.0
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps % self.growth_interval == 0:
                self.scale *= 2.0

    def state_dict(self):
        return {"scale": self.scale, "good_steps": self.good_steps}

    def load_state_dict(self, d):
        self.scale, self.good_steps = d["scale"], d["good_steps"]
