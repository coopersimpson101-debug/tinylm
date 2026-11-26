import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tinyinfer_dir():
    """Location of a tinyinfer checkout (env TINYINFER_DIR, ./tinyinfer or ../tinyinfer)."""
    for c in (os.environ.get("TINYINFER_DIR"), ROOT / "tinyinfer", ROOT.parent / "tinyinfer"):
        if c and (Path(c) / "python" / "tinyinfer_ref").exists():
            return Path(c)
    return None


def make_stories(n=300, seed=0):
    r = random.Random(seed)
    names = ["Tom", "Lily", "Ben", "Sue", "Max", "Mia"]
    things = ["ball", "kite", "dog", "cake", "box", "tree", "boat"]
    feels = ["happy", "sad", "excited", "tired", "scared"]
    out = []
    for _ in range(n):
        a, b = r.sample(names, 2)
        t = r.choice(things)
        s = (f"Once upon a time, there was a kid named {a}. {a} had a {r.choice(feels)} {t}. "
             f"One day, {a} and {b} went to the park to play with the {t}. "
             f"\"Look at my {t}!\" said {a}. {b} was {r.choice(feels)} and they played all day. "
             f"At the end of the day, {a} and {b} went home. They were {r.choice(feels)}.")
        out.append(s)
    return out


@pytest.fixture(scope="session")
def stories():
    return make_stories()


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory, stories):
    """A tokenizer + train/valid shards built from synthetic stories."""
    from tinylm import bpe, shards
    d = tmp_path_factory.mktemp("data")
    counts = bpe.count_words(stories)
    merges, _ = bpe.train(counts, 512)
    tok = bpe.Tokenizer(merges)
    tok.save(str(d / "tokenizer.json"))
    for name, docs in (("train", stories[:260]), ("valid", stories[260:])):
        w = shards.ShardWriter(str(d), name, tok.vocab_size, shard_tokens=8000)
        for s in docs:
            w.add(tok.encode(s) + [tok.eos_id])
        w.flush()
    return d, tok
