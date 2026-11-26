"""Qwen2-architecture decoder-only transformer, written from tensor primitives.

Only ``torch.Tensor`` operations and autograd are used: no ``nn.Transformer``,
``scaled_dot_product_attention``, ``nn.LayerNorm`` or ``nn.functional.rms_norm``.

Per block (pre-norm residual)::

    h  = x + Wo . Attn(RoPE(Wq n + bq), RoPE(Wk n + bk), Wv n + bv),   n = RMSNorm(x)
    x' = h + Wdown . (SiLU(Wgate m) * (Wup m)),                        m = RMSNorm(h)
    logits = RMSNorm(x_L) . E^T                                          (tied embeddings)

* RMSNorm (Zhang & Sennrich 2019): x / sqrt(mean(x^2) + eps) * g.
* RoPE (Su et al. 2021) in the "rotate-half" (GPT-NeoX) layout used by Qwen2:
  the pair (x_i, x_{i+d/2}) is rotated by angle p * theta^(-2i/d). This is the
  layout tinyinfer's C kernel implements, so weights export without permutation.
* Grouped-query attention (Ainslie et al. 2023): query head h reads key/value
  head floor(h / (H / H_kv)).
* SwiGLU feed-forward (Shazeer 2020).
* Bias on the Q, K and V projections only (as in Qwen2).
"""
import math

import torch
import torch.nn as nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def rope_tables(head_dim: int, max_len: int, theta: float, device=None):
    """cos/sin tables of shape (max_len, head_dim/2), computed in float64 then cast."""
    inv = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    ang = torch.arange(max_len, dtype=torch.float64)[:, None] * inv[None, :]
    return ang.cos().float().to(device), ang.sin().float().to(device)


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd); cos/sin: (T, hd/2)."""
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.wq = nn.Linear(c.dim, c.n_heads * c.head_dim, bias=c.qkv_bias)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=c.qkv_bias)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=c.qkv_bias)
        self.wo = nn.Linear(c.n_heads * c.head_dim, c.dim, bias=False)

    def forward(self, x, cos, sin, mask, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # incremental decoding: (k, v) of earlier positions
            if cache.get("k") is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        g = self.nh // self.nkv
        if g > 1:
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
        att = att.float() + mask
        att = torch.softmax(att, dim=-1).to(v.dtype)
        y = (att @ v).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.up = nn.Linear(c.dim, c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, c.dim, bias=False)

    def forward(self, x):
        g = self.gate(x)
        return self.down(g * torch.sigmoid(g) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(c.dim, c.rms_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.dim, c.rms_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, mask, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask, cache)
        return x + self.mlp(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        c.validate()
        self.cfg = c
        self.tok_embd = nn.Embedding(c.vocab_size, c.dim)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim, c.rms_eps)
        self.output = None if c.tied_embeddings else nn.Linear(c.dim, c.vocab_size, bias=False)
        cos, sin = rope_tables(c.head_dim, c.max_seq_len, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02):
        """GPT-2 style init: N(0, 0.02); residual output projections scaled by 1/sqrt(2L)."""
        for name, p in self.named_parameters():
            if p.ndim == 1:
                nn.init.ones_(p) if name.endswith("norm.weight") else nn.init.zeros_(p)
            elif name.endswith("wo.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, 0.0, std / math.sqrt(2 * self.cfg.n_layers))
            else:
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets=None, start_pos: int = 0, caches=None):
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start_pos + T} > max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_embd(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        S = start_pos + T
        mask = torch.full((T, S), float("-inf"), device=idx.device).triu(start_pos + 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask, None if caches is None else caches[i])
        x = self.norm(x)
        w = self.tok_embd.weight if self.output is None else self.output.weight
        logits = x @ w.to(x.dtype).t()
        if targets is None:
            return logits
        loss = cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40,
                 eos_id: int | None = None, generator=None):
        """Sample with a per-layer KV cache. idx: (1, T) prompt."""
        caches = [dict() for _ in self.layers]
        logits = self(idx, caches=caches)[:, -1]
        out = idx
        for _ in range(max_new_tokens):
            if out.shape[1] >= self.cfg.max_seq_len:
                break
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                lg = logits.float() / temperature
                if top_k:
                    kth = torch.topk(lg, min(top_k, lg.shape[-1])).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and int(nxt) == eos_id:
                break
            logits = self(nxt, start_pos=out.shape[1] - 1, caches=caches)[:, -1]
        return out


def cross_entropy(logits, targets):
    """Mean token negative log-likelihood, -log softmax(z)_y, via a stable log-sum-exp."""
    z = logits.float().reshape(-1, logits.shape[-1])
    y = targets.reshape(-1)
    lse = torch.logsumexp(z, dim=-1)
    return (lse - z.gather(1, y[:, None]).squeeze(1)).mean()


def token_logprobs(model, idx):
    """log p(idx[t] | idx[<t]) for t >= 1; shape (B, T-1)."""
    logits = model(idx[:, :-1]).float()
    lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return lp.gather(2, idx[:, 1:, None]).squeeze(2)
