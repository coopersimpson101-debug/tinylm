"""Byte-level BPE: trainer, encoder and decoder (GPT-2 / Qwen2 flavour).

Training (Sennrich et al. 2016; byte-level variant of Radford et al. 2019):

1. pre-tokenise the corpus (``tinylm.pretok``) and count each distinct
   pre-token ("word");
2. write every word as a sequence of byte symbols (ids 0..255);
3. repeatedly find the most frequent adjacent symbol pair (a, b), add the
   token ``ab`` with the next id and record the merge ``(a, b)``; rewrite
   every word containing the pair.

A naive implementation recounts all pairs after every merge (O(merges x
corpus)). Here pair counts are kept incrementally: an index pair -> {words
containing it} means only the words that actually contain the merged pair are
touched, and a max-heap with lazy deletion finds the best pair in O(log P).
Ties are broken towards the pair with the smaller ids, so training is
deterministic.

Encoding applies merges by rank (lowest rank first), which reproduces the
segmentation seen at training time. Tokens are stored as GPT-2
``bytes_to_unicode`` strings so the vocabulary and merge list can be written
straight into a GGUF file (``tokenizer.ggml.tokens`` / ``tokenizer.ggml.merges``)
and read by tinyinfer's C tokenizer.
"""
import heapq
import json
from collections import Counter

from .pretok import pretokenize_fast

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable code point table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE2U = bytes_to_unicode()
U2BYTE = {v: k for k, v in BYTE2U.items()}


def count_words(texts, counter: Counter | None = None) -> Counter:
    counter = counter if counter is not None else Counter()
    for t in texts:
        counter.update(pretokenize_fast(t))
    return counter


def train(word_counts: dict, vocab_size: int, specials=SPECIALS, min_freq: int = 2, verbose=False):
    """Train byte-level BPE. Returns (merges as list of (id_a, id_b), token byte strings)."""
    n_merges = vocab_size - 256 - len(specials)
    if n_merges < 0:
        raise ValueError("vocab_size too small")
    words = []   # list of symbol-id lists
    freqs = []
    for w, c in word_counts.items():
        b = w.encode("utf-8")
        if b:
            words.append(list(b))
            freqs.append(c)
    token_bytes = [bytes([i]) for i in range(256)]
    pair_count: dict = {}
    where: dict = {}
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pair_count[p] = pair_count.get(p, 0) + f
            where.setdefault(p, set()).add(wi)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) < n_merges and heap:
        negc, p = heapq.heappop(heap)
        c = pair_count.get(p, 0)
        if c != -negc:              # stale entry (a fresher one was pushed when c changed)
            continue
        if c < min_freq:
            break
        a, b = p
        new_id = len(token_bytes)
        token_bytes.append(token_bytes[a] + token_bytes[b])
        merges.append(p)
        changed = {}
        for wi in where.pop(p, ()):
            w = words[wi]
            f = freqs[wi]
            # remove old pairs of this word
            for q in zip(w, w[1:]):
                pair_count[q] -= f
                changed[q] = True
                s = where.get(q)
                if s is not None:
                    s.discard(wi)
            out = []
            i = 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for q in zip(out, out[1:]):
                pair_count[q] = pair_count.get(q, 0) + f
                changed[q] = True
                where.setdefault(q, set()).add(wi)
        pair_count.pop(p, None)
        for q in changed:
            cq = pair_count.get(q, 0)
            if cq > 0:
                heapq.heappush(heap, (-cq, q))
            else:
                pair_count.pop(q, None)
                where.pop(q, None)
        if verbose and len(merges) % 500 == 0:
            print(f"  merge {len(merges)}/{n_merges}: {token_bytes[new_id]!r} x{c}", flush=True)
    return merges, token_bytes


class Tokenizer:
    """Byte-level BPE tokenizer with GGUF-compatible string vocabulary."""

    def __init__(self, merges, specials=SPECIALS):
        self.merges = [tuple(m) for m in merges]
        tb = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            tb.append(tb[a] + tb[b])
        self.token_bytes = tb
        self.specials = list(specials)
        self.tokens = ["".join(BYTE2U[x] for x in t) for t in tb] + self.specials
        self.types = [1] * len(tb) + [3] * len(self.specials)
        self.vocab = {}
        for i, t in enumerate(self.tokens):
            self.vocab.setdefault(t, i)       # first wins (same rule as tinyinfer's C loader)
        self.special_ids = {s: len(tb) + k for k, s in enumerate(self.specials)}
        self.eos_id = self.special_ids["<|endoftext|>"]
        self.ranks = {(self.tokens[a], self.tokens[b]): r for r, (a, b) in enumerate(self.merges)}
        self._cache: dict = {}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def merge_strings(self) -> list[str]:
        return [self.tokens[a] + " " + self.tokens[b] for a, b in self.merges]

    def _bpe(self, word: str) -> list[int]:
        ids = self._cache.get(word)
        if ids is not None:
            return ids
        syms = [BYTE2U[b] for b in word.encode("utf-8")]
        ranks = self.ranks
        while len(syms) > 1:
            best, best_r = -1, None
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best, best_r = i, r
            if best < 0:
                break
            syms[best:best + 2] = [syms[best] + syms[best + 1]]
        ids = [self.vocab[s] for s in syms]
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        out: list[int] = []
        if allow_special and "<|" in text:
            i = seg = 0
            while i < len(text):
                hit = None
                if text.startswith("<|", i):
                    for s in sorted(self.specials, key=len, reverse=True):
                        if text.startswith(s, i):
                            hit = s
                            break
                if hit:
                    out += self.encode(text[seg:i], False)
                    out.append(self.special_ids[hit])
                    i += len(hit)
                    seg = i
                else:
                    i += 1
            return out + self.encode(text[seg:], False)
        for w in pretokenize_fast(text):
            out.extend(self._bpe(w))
        return out

    def decode_bytes(self, ids) -> bytes:
        nb = len(self.token_bytes)
        return b"".join(self.token_bytes[i] if i < nb else self.tokens[i].encode() for i in ids)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        json.dump({"type": "byte-level-bpe", "pretokenizer": "qwen2", "merges": self.merges,
                   "specials": self.specials}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["merges"], d.get("specials", SPECIALS))
