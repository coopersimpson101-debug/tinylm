"""Pre-tokeniser, BPE, cleaning, dedup and shards."""
import random
from collections import Counter

import numpy as np
import pytest

from tinylm import bpe, clean, dedup, shards
from tinylm.pretok import pretokenize, pretokenize_fast

QWEN2 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
         r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
SAMPLES = ["Hello world!", "  leading and trailing  ", "I'm sure they'll say we've won 1234 times.",
           "line\n\n  indented\r\n\ttab", "Café naïve über straße добрый день καλημέρα",
           "你好，世界。 こんにちは 한국어 😀 🚀", "x=1+2; y = f(x)  # comment\n", "''quoted'' 'S 'RE",
           "A B C   \n", "!!!???...,,,", "$100.50 and 3.14159"]


def test_pretok_matches_reference_regex():
    regex = pytest.importorskip("regex")
    for s in SAMPLES:
        assert pretokenize(s) == regex.findall(QWEN2, s), s
    r = random.Random(0)
    alpha = list("ab Z09'\n\r\t .,!?-é字😀 ") + ["'s", "'LL", "'re"]
    for _ in range(3000):
        s = "".join(r.choice(alpha) for _ in range(r.randint(0, 25)))
        assert pretokenize(s) == regex.findall(QWEN2, s), repr(s)


def test_pretok_fast_path_agrees(stories):
    r = random.Random(1)
    alpha = list("abcXYZ019 '\n\r\t\x0b\x0c\x1c.,!?-()") + ["'s", "'ll", "'Ve"]
    texts = SAMPLES + stories[:50] + ["".join(r.choice(alpha) for _ in range(40)) for _ in range(2000)]
    for s in texts:
        assert pretokenize_fast(s) == pretokenize(s), repr(s)
        assert "".join(pretokenize(s)) == s


def naive_bpe(word_counts, vocab_size, n_special=3):
    """Textbook BPE: recount every pair after every merge (reference for the fast trainer)."""
    words = {tuple(w.encode()): c for w, c in word_counts.items()}
    merges, nxt = [], 256
    while len(merges) < vocab_size - 256 - n_special:
        pairs = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < 2:
            break
        merges.append(best)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(nxt)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
        nxt += 1
    return merges


def test_incremental_trainer_equals_naive(stories):
    counts = bpe.count_words(stories[:80] + ["aaabdaaabac " * 3, "the theme then there"])
    fast, _ = bpe.train(counts, 400)
    assert [tuple(m) for m in fast] == naive_bpe(counts, 400)


def test_classic_example_first_merge():
    merges, tb = bpe.train({"aaabdaaabac": 1, "aa": 5}, 256 + 3 + 2)
    assert tb[256] == b"aa"


def test_tokenizer_roundtrip_and_specials(tiny_data, tmp_path):
    _, tok = tiny_data
    for s in SAMPLES + ["Once upon a time, Tom had a ball."]:
        ids = tok.encode(s)
        assert tok.decode(ids) == s
        assert all(0 <= i < tok.vocab_size for i in ids)
    ids = tok.encode("a<|endoftext|>b")
    assert tok.eos_id in ids and tok.decode(ids) == "a<|endoftext|>b"
    assert tok.eos_id not in tok.encode("a<|endoftext|>b", allow_special=False)
    tok.save(str(tmp_path / "t.json"))
    t2 = bpe.Tokenizer.load(str(tmp_path / "t.json"))
    assert t2.tokens == tok.tokens and t2.encode("Tom and Lily") == tok.encode("Tom and Lily")
    # compression on in-domain text
    assert len(tok.encode(" ".join(SAMPLES[:1] * 0) + "Once upon a time, there was a kid")) < 20


def test_merge_strings_are_gpt2_bytelevel(tiny_data):
    _, tok = tiny_data
    assert tok.tokens[ord("A")] == "A" and tok.tokens[ord(" ")] == "Ġ"
    for m in tok.merge_strings[:50]:
        a, b = m.split(" ")
        assert a + b in tok.vocab


def test_normalize_and_quality():
    t = clean.normalize("“Hi,” she said—it’s fine…  ok​\r\n\n\n\nEnd ")
    assert t == '"Hi," she said-it\'s fine... ok\n\nEnd'
    good = "Once upon a time there was a little dog. The dog liked to run in the park with his friend."
    assert clean.quality_check(good * 2) is None
    assert clean.quality_check("too short") == "length"
    assert clean.quality_check("1234 5678 9999 0000 " * 20) == "alpha"
    assert clean.quality_check("Der Hund lief schnell nach Hause und spielte mit dem Ball. " * 4) == "english"
    assert clean.quality_check(("the dog ran " * 60)) == "repetition"
    assert clean.quality_check(good + " <|endoftext|>") == "markup"


def test_split_documents(tmp_path):
    p = tmp_path / "raw.txt"
    p.write_text("one<|endoftext|>two<|endoftext|>thr", encoding="utf-8")
    assert list(clean.split_documents(str(p), chunk=4)) == ["one", "two", "thr"]
    assert list(clean.split_documents(str(p), truncated=True, chunk=5)) == ["one", "two"]


def test_minhash_estimates_jaccard(stories):
    mh = dedup.MinHasher(n_perm=256, seed=3)
    a, b = stories[0], stories[0].replace("park", "beach").replace("home", "school")
    sa, sb = set(mh.shingles(a).tolist()), set(mh.shingles(b).tolist())
    true = dedup.jaccard(sa, sb)
    est = float((mh.signature(a) == mh.signature(b)).mean())
    assert abs(est - true) < 0.1
    assert float((mh.signature(a) == mh.signature(stories[1])).mean()) < true


def test_near_duplicates_found_and_unrelated_kept(stories):
    mh = dedup.MinHasher(n_perm=64, seed=1)
    base = [" ".join(s.split() * 3) for s in stories[:40]]  # longer docs, distinct
    docs = base + [base[3] + " The end.", base[7].replace("Once", "Once,")]
    sigs = np.stack([mh.signature(d) for d in docs])
    drop, _ = dedup.near_duplicates(sigs, bands=16, rows=4, threshold=0.8)
    assert set(drop.tolist()) == {40, 41}
    assert dedup.exact_key("Hello  World") == dedup.exact_key("hello world")


def test_shards_roundtrip_and_sampling(tmp_path):
    w = shards.ShardWriter(str(tmp_path), "train", 1000, shard_tokens=100)
    data = np.arange(250) % 1000
    w.add(data[:120])
    w.add(data[120:])
    w.flush()
    assert w.n_shards == 3 and w.total == 250
    got = np.concatenate([shards.read_shard(str(tmp_path / f"train_{i:03d}.bin")) for i in range(3)])
    assert np.array_equal(got, data)
    ds = shards.TokenDataset(str(tmp_path / "train_*.bin"), seq_len=8, seed=5)
    x1, y1 = ds.batch(7, 4)
    x2, y2 = ds.batch(7, 4)
    assert np.array_equal(x1, x2) and np.array_equal(x1[:, 1:], y1[:, :-1])
    assert not np.array_equal(ds.batch(8, 4)[0], x1)
    r0 = shards.TokenDataset(str(tmp_path / "train_*.bin"), 8, seed=5, rank=0, world=2).batch(7, 4)[0]
    r1 = shards.TokenDataset(str(tmp_path / "train_*.bin"), 8, seed=5, rank=1, world=2).batch(7, 4)[0]
    assert not np.array_equal(r0, r1)
    with pytest.raises(ValueError):
        shards.write_shard(str(tmp_path / "bad.bin"), np.array([70000]), 70001)
