"""Sharded uint16 token files and a random-window batch sampler.

Shard layout (little endian)::

    bytes 0..1023   header: int32 magic=20261126, int32 version=1,
                    int64 n_tokens, int32 vocab_size, zero padding
    bytes 1024..    n_tokens x uint16 token ids

Documents are concatenated, each followed by the end-of-text token, so a
training window may span a document boundary (standard GPT pretraining).
"""
import glob
import os
import struct

import numpy as np

MAGIC = 20261126
HEADER = 1024


def write_shard(path: str, tokens: np.ndarray, vocab_size: int):
    tokens = np.asarray(tokens)
    if vocab_size > 65536 or (tokens.size and int(tokens.max()) >= vocab_size):
        raise ValueError("token id out of range for uint16 shard")
    hdr = struct.pack("<iiqi", MAGIC, 1, int(tokens.size), vocab_size).ljust(HEADER, b"\0")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr)
        f.write(tokens.astype("<u2").tobytes())
    os.replace(tmp, path)


def read_shard(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, ver, n, vocab = struct.unpack("<iiqi", f.read(20))
    if magic != MAGIC or ver != 1:
        raise ValueError(f"{path}: not a tinylm shard")
    return np.memmap(path, dtype="<u2", mode="r", offset=HEADER, shape=(n,))


class ShardWriter:
    """Accumulates documents and flushes fixed-size shards."""

    def __init__(self, out_dir: str, prefix: str, vocab_size: int, shard_tokens: int = 32 << 20):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.vocab = out_dir, prefix, vocab_size
        self.shard_tokens = shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.fill = 0
        self.n_shards = 0
        self.total = 0

    def add(self, ids):
        ids = np.asarray(ids, dtype=np.uint16)
        while ids.size:
            take = min(ids.size, self.shard_tokens - self.fill)
            self.buf[self.fill:self.fill + take] = ids[:take]
            self.fill += take
            ids = ids[take:]
            if self.fill == self.shard_tokens:
                self.flush()

    def flush(self):
        if self.fill:
            p = os.path.join(self.out_dir, f"{self.prefix}_{self.n_shards:03d}.bin")
            write_shard(p, self.buf[:self.fill], self.vocab)
            self.n_shards += 1
            self.total += self.fill
            self.fill = 0


class TokenDataset:
    """Samples (x, y) windows uniformly at random from a set of shards.

    ``rank`` / ``world`` give each data-parallel worker an independent stream;
    the sampler state is a single integer (``step``), so resuming is exact.
    """

    def __init__(self, pattern: str, seq_len: int, seed: int = 0, rank: int = 0, world: int = 1):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        self.shards = [read_shard(f) for f in files]
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.seq_len = seq_len
        self.seed, self.rank, self.world = seed, rank, world
        self.n_tokens = int(self.sizes.sum())

    def batch(self, step: int, batch_size: int):
        rng = np.random.default_rng([self.seed, self.rank, self.world, step])
        T = self.seq_len
        w = np.maximum(self.sizes - T - 1, 0).astype(np.float64)
        shard_idx = rng.choice(len(self.shards), size=batch_size, p=w / w.sum())
        x = np.empty((batch_size, T + 1), dtype=np.int64)
        for i, s in enumerate(shard_idx):
            off = int(rng.integers(0, self.sizes[s] - T - 1))
            x[i] = self.shards[s][off:off + T + 1]
        return x[:, :-1], x[:, 1:]

    def sequential(self, batch_size: int, max_batches: int | None = None):
        """Non-overlapping windows in order (for deterministic evaluation)."""
        T = self.seq_len
        rows = []
        n = 0
        for s in self.shards:
            for off in range(0, len(s) - T - 1, T):
                rows.append(np.asarray(s[off:off + T + 1], dtype=np.int64))
                if len(rows) == batch_size:
                    x = np.stack(rows)
                    rows = []
                    yield x[:, :-1], x[:, 1:]
                    n += 1
                    if max_batches and n >= max_batches:
                        return
        if rows:
            x = np.stack(rows)
            yield x[:, :-1], x[:, 1:]
