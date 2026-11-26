"""End-to-end data pipeline: raw text -> cleaned, deduplicated, tokenised shards.

    python -m tinylm.prepare --train data/raw/tinystories-train.txt --train-truncated \\
        --valid data/raw/tinystories-valid.txt --out data/ts4k --vocab 4096

Stages (each parallelised over document batches with ``multiprocessing``):

1. split on ``<|endoftext|>``, ``clean.normalize`` + ``clean.quality_check``;
2. exact dedup (BLAKE2b of normalised text) then MinHash-LSH near dedup;
   validation documents that exactly match a training document are removed;
3. train the byte-level BPE tokenizer on the first ``--tok-mb`` MB of the
   surviving training text;
4. encode every document, append ``<|endoftext|>``, and write uint16 shards.

A machine-readable ``report.json`` records every count.
"""
import argparse
import json
import os
import shutil
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

from . import bpe, clean, dedup, shards

_hasher = None
_tok = None


def _init_worker(tok_path=None):
    global _hasher, _tok
    _hasher = dedup.MinHasher(n_perm=64, shingle=5, seed=1234)
    if tok_path:
        _tok = bpe.Tokenizer.load(tok_path)


def _clean_batch(args):
    idx, docs, tmp_dir, split = args
    reasons = Counter()
    kept, keys, sigs = [], [], []
    for d in docs:
        t = clean.normalize(d)
        r = clean.quality_check(t)
        if r:
            reasons[r] += 1
            continue
        kept.append(t)
        keys.append(dedup.exact_key(t))
        sigs.append(_hasher.signature(t))
    path = os.path.join(tmp_dir, f"{split}_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\0".join(kept))
    sig = np.stack(sigs) if sigs else np.zeros((0, 64), np.uint32)
    return idx, path, len(docs), reasons, np.array(keys, dtype=np.uint64), sig


def _read_batch(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        s = f.read()
    return s.split("\0") if s else []


def _count_batch(args):
    path, keep = args
    docs = [d for d, k in zip(_read_batch(path), keep) if k]
    return bpe.count_words(docs), sum(len(d) for d in docs)


def _encode_batch(args):
    path, keep = args
    out = []
    eos = _tok.eos_id
    for d, k in zip(_read_batch(path), keep):
        if k:
            out.extend(_tok.encode(d, allow_special=False))
            out.append(eos)
    return np.asarray(out, dtype=np.uint16)


def _batches(path, truncated, size):
    buf = []
    i = 0
    for d in clean.split_documents(path, truncated=truncated):
        buf.append(d)
        if len(buf) == size:
            yield i, buf
            i += 1
            buf = []
    if buf:
        yield i, buf


def clean_split(pool, path, split, tmp, truncated, batch_size, log):
    t0 = time.time()
    results = []
    jobs = ((i, b, tmp, split) for i, b in _batches(path, truncated, batch_size))
    for r in pool.imap(_clean_batch, jobs):
        results.append(r)
        if len(results) % 20 == 0:
            log(f"  [{split}] cleaned {len(results)} batches")
    results.sort(key=lambda r: r[0])
    n_in = sum(r[2] for r in results)
    reasons = sum((r[3] for r in results), Counter())
    keys = np.concatenate([r[4] for r in results])
    sigs = np.concatenate([r[5] for r in results])
    sizes = [len(r[4]) for r in results]
    files = [r[1] for r in results]
    log(f"  [{split}] {n_in} docs, {len(keys)} pass filters ({dict(reasons)}) in {time.time() - t0:.0f}s")
    return files, sizes, n_in, reasons, keys, sigs


def run(a):
    log = lambda m: print(m, flush=True)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    report = {"inputs": {"train": os.path.basename(a.train), "valid": os.path.basename(a.valid)}}
    T0 = time.time()
    with Pool(a.workers, initializer=_init_worker) as pool:
        tr = clean_split(pool, a.train, "train", tmp, a.train_truncated, a.batch_docs, log)
        va = clean_split(pool, a.valid, "valid", tmp, False, a.batch_docs, log)

    def dedup_split(name, files, sizes, n_in, reasons, keys, sigs, banned=None):
        t0 = time.time()
        keep = np.ones(len(keys), dtype=bool)
        seen = set()
        n_exact = n_leak = 0
        for i, k in enumerate(keys.tolist()):
            if banned is not None and k in banned:
                keep[i] = False
                n_leak += 1
            elif k in seen:
                keep[i] = False
                n_exact += 1
            else:
                seen.add(k)
        idx = np.flatnonzero(keep)
        drop, n_cand = dedup.near_duplicates(sigs[idx], bands=16, rows=4, threshold=a.near_threshold)
        keep[idx[drop]] = False
        log(f"  [{name}] exact dups {n_exact}, near dups {len(drop)} ({n_cand} LSH candidates), "
            f"leaked-from-train {n_leak}; kept {int(keep.sum())} in {time.time() - t0:.0f}s")
        masks, o = [], 0
        for s in sizes:
            masks.append(keep[o:o + s].tolist())
            o += s
        report[name] = {"documents_in": n_in, "filtered": dict(reasons), "passed_filters": len(keys),
                        "exact_duplicates": n_exact, "near_duplicates": int(len(drop)),
                        "lsh_candidates": int(n_cand), "exact_overlap_with_train": n_leak,
                        "documents_kept": int(keep.sum())}
        return list(zip(files, masks))

    train_jobs = dedup_split("train", *tr)
    train_keys = set(tr[4].tolist())
    valid_jobs = dedup_split("valid", *va, banned=train_keys)
    del tr, va, train_keys

    tok_path = os.path.join(a.out, "tokenizer.json")
    if a.tokenizer:
        shutil.copy(a.tokenizer, tok_path)
        log(f"using existing tokenizer {a.tokenizer}")
    else:
        t0 = time.time()
        counts = Counter()
        n_batches = max(1, int(a.tok_mb * 1e6 / (a.batch_docs * 800)))  # ~800 chars per story
        jobs = train_jobs[:n_batches]
        chars = 0
        with Pool(a.workers, initializer=_init_worker) as pool:
            for c, n in pool.imap(_count_batch, jobs):
                counts.update(c)
                chars += n
        log(f"tokenizer corpus: {chars / 1e6:.1f} MB, {len(counts)} distinct pre-tokens")
        merges, _ = bpe.train(counts, a.vocab, verbose=True)
        tok = bpe.Tokenizer(merges)
        tok.save(tok_path)
        report["tokenizer"] = {"vocab_size": tok.vocab_size, "merges": len(merges),
                               "train_chars": chars, "distinct_pretokens": len(counts),
                               "seconds": round(time.time() - t0, 1)}
        log(f"trained BPE: vocab {tok.vocab_size} in {time.time() - t0:.0f}s")
    tok = bpe.Tokenizer.load(tok_path)

    with Pool(a.workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        for name, jobs in (("train", train_jobs), ("valid", valid_jobs)):
            t0 = time.time()
            w = shards.ShardWriter(a.out, name, tok.vocab_size, a.shard_tokens)
            for arr in pool.imap(_encode_batch, jobs):
                w.add(arr)
            w.flush()
            report[name]["tokens"] = int(w.total)
            report[name]["shards"] = w.n_shards
            log(f"  [{name}] {w.total} tokens in {w.n_shards} shards ({time.time() - t0:.0f}s)")
    if not a.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    report["seconds"] = round(time.time() - T0, 1)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    log(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--train-truncated", action="store_true",
                    help="train file is a byte-range prefix: drop the trailing partial document")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tok-mb", type=float, default=60.0, help="MB of text used to train BPE")
    ap.add_argument("--tokenizer", help="reuse an existing tokenizer.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--batch-docs", type=int, default=4000)
    ap.add_argument("--near-threshold", type=float, default=0.8)
    ap.add_argument("--shard-tokens", type=int, default=32 << 20)
    ap.add_argument("--keep-tmp", action="store_true")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
