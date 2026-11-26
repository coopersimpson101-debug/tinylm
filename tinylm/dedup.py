"""Exact and near-duplicate detection (MinHash + LSH), implemented from scratch.

Exact duplicates: a 64-bit BLAKE2b digest of the lower-cased,
whitespace-collapsed text.

Near duplicates (Broder 1997; Leskovec, Rajaraman & Ullman, *Mining of Massive
Datasets*, ch. 3):

* a document is the set S of its word 5-gram shingles;
* the Jaccard similarity J(A, B) = |A n B| / |A u B|;
* for a random hash h, P[min h(A) = min h(B)] = J(A, B), so a signature of
  ``n_perm`` independent min-hashes estimates J by the fraction of equal slots.
  The hash family is multiply-shift, h_i(x) = (a_i x + b_i mod 2^64) >> 32 with
  a_i odd (Dietzfelbinger et al. 1997), vectorised in NumPy;
* LSH banding: split the signature into ``bands`` bands of ``rows`` rows; two
  documents become candidates if any band matches exactly. The probability of
  becoming a candidate is 1 - (1 - J^r)^b, an S-curve with threshold
  ~ (1/b)^(1/r) (0.5 for 16 x 4);
* candidates are verified with the full signature (estimated J >= threshold),
  clustered with union-find, and all but the first document of every cluster
  are dropped.
"""
import hashlib
import zlib

import numpy as np

_MASK32 = np.uint64(0xFFFFFFFF)


def exact_key(text: str) -> int:
    norm = " ".join(text.lower().split())
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, n_perm: int = 64, shingle: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_perm = n_perm
        self.k = shingle
        self.a = (rng.integers(0, 2**63, n_perm, dtype=np.uint64) << np.uint64(1)) | np.uint64(1)
        self.b = rng.integers(0, 2**63, n_perm, dtype=np.uint64)
        self._word_cache: dict = {}

    def _word_hashes(self, text: str) -> np.ndarray:
        wc = self._word_cache
        out = []
        for w in text.lower().split():
            h = wc.get(w)
            if h is None:
                h = zlib.crc32(w.encode("utf-8"))
                if len(wc) < 1_000_000:
                    wc[w] = h
            out.append(h)
        return np.asarray(out, dtype=np.uint64)

    def shingles(self, text: str) -> np.ndarray:
        """32-bit hashes of word k-grams (polynomial rolling combination)."""
        w = self._word_hashes(text)
        k = self.k
        if len(w) == 0:
            return np.zeros(1, dtype=np.uint64)
        if len(w) < k:
            k = len(w)
        n = len(w) - k + 1
        acc = np.zeros(n, dtype=np.uint64)
        mult = np.uint64(0x01000193)
        for j in range(k):
            acc = (acc * mult + w[j:j + n]) & _MASK32
        return np.unique(acc)

    def signature(self, text: str) -> np.ndarray:
        s = self.shingles(text)
        h = (s[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)
        return h.min(axis=0).astype(np.uint32)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:  # keep the smaller (earlier) index as the root
            if rx < ry:
                self.p[ry] = rx
            else:
                self.p[rx] = ry


def band_keys(sigs: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """(N, bands) uint64 keys; equal key == equal band (up to 2^-64 collisions)."""
    n, m = sigs.shape
    assert bands * rows <= m
    s = sigs[:, :bands * rows].astype(np.uint64).reshape(n, bands, rows)
    key = np.zeros((n, bands), dtype=np.uint64)
    for r in range(rows):
        key = key * np.uint64(0x9E3779B97F4A7C15) + s[:, :, r] + np.uint64(r + 1)
        key ^= key >> np.uint64(29)
    return key


def near_duplicates(sigs: np.ndarray, bands: int = 16, rows: int = 4, threshold: float = 0.8):
    """Indices of documents to drop (every cluster keeps its earliest member)."""
    n = len(sigs)
    if n < 2:
        return np.zeros(0, dtype=np.int64), 0
    keys = band_keys(sigs, bands, rows)
    uf = UnionFind(n)
    n_candidates = 0
    for b in range(bands):
        order = np.argsort(keys[:, b], kind="stable")
        kb = keys[order, b]
        starts = np.flatnonzero(np.r_[True, kb[1:] != kb[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            group = order[s:e]
            rep = group.min()
            others = group[group != rep]
            n_candidates += len(others)
            sim = (sigs[others] == sigs[rep]).mean(axis=1)
            for d in others[sim >= threshold]:
                uf.union(int(rep), int(d))
    roots = np.array([uf.find(i) for i in range(n)])
    drop = np.flatnonzero(roots != np.arange(n))
    return drop, n_candidates
