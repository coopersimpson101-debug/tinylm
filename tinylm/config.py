"""Model configurations (Qwen2 architecture) and parameter counting."""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_dim: int = 1408
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    qkv_bias: bool = True
    tied_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def validate(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        return self

    def n_params(self, embeddings: bool = True) -> int:
        d, hd, f, v = self.dim, self.head_dim, self.ffn_dim, self.vocab_size
        q, kv = self.n_heads * hd, self.n_kv_heads * hd
        attn = d * q + 2 * d * kv + q * d + (q + 2 * kv if self.qkv_bias else 0)
        layer = attn + 3 * d * f + 2 * d
        total = self.n_layers * layer + d
        if embeddings:
            total += v * d * (1 if self.tied_embeddings else 2)
        return total

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Training FLOPs/token: 6N (matmuls, fwd+bwd) + 12 L T d (attention scores)."""
        T = seq_len or self.max_seq_len
        return 6 * self.n_params(embeddings=False) + 6 * self.vocab_size * self.dim + \
            12 * self.n_layers * T * self.dim

    def to_dict(self):
        return asdict(self)


PRESETS = {
    # unit tests / CI
    "smoke": ModelConfig(vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=160,
                         max_seq_len=128),
    # speculative-decoding draft model
    "tl-3m": ModelConfig(dim=192, n_layers=6, n_heads=6, n_kv_heads=2, ffn_dim=512),
    # "10M (smoke)" config from the brief: trains on a CPU in minutes
    "tl-10m": ModelConfig(dim=320, n_layers=8, n_heads=8, n_kv_heads=4, ffn_dim=864),
    # target model trained in this repo
    "tl-25m": ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, ffn_dim=1408),
    # home-machine runs (FineWeb-Edu, 32k vocab) -- see README, not trained here
    "tl-125m": ModelConfig(vocab_size=32768, dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
                           ffn_dim=2944, max_seq_len=1024),
    "tl-300m": ModelConfig(vocab_size=32768, dim=1024, n_layers=24, n_heads=16, n_kv_heads=4,
                           ffn_dim=2816, max_seq_len=1024),
}


def get_config(name: str, **overrides) -> ModelConfig:
    import copy
    c = copy.deepcopy(PRESETS[name])
    for k, v in overrides.items():
        setattr(c, k, v)
    return c.validate()


if __name__ == "__main__":
    for k, c in PRESETS.items():
        print(f"{k:8s} params={c.n_params() / 1e6:7.2f}M non-emb={c.n_params(False) / 1e6:7.2f}M "
              f"flops/tok={c.flops_per_token() / 1e6:.0f}M")
