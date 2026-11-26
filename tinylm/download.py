"""Download openly licensed pretraining corpora from the Hugging Face Hub.

Only the Python standard library is used (urllib). Large files are fetched
with an HTTP ``Range`` request so a bounded prefix can be taken (the
TinyStoriesV2 train file is 2.2 GB; a 1 GB prefix is plenty for a home run).

Every download is recorded in ``<out>/SOURCES.json`` with URL, byte range,
size, SHA-256 and licence, so the provenance of each shard is auditable.

    python -m tinylm.download tinystories --out data/raw --max-mb 1000
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HF = "https://huggingface.co/datasets"

CORPORA = {
    "tinystories": {
        "repo": "roneneldan/TinyStories",
        "license": "CDLA-Sharing-1.0",
        "paper": "Eldan & Li 2023, arXiv:2305.07759",
        "files": {
            "train": "TinyStoriesV2-GPT4-train.txt",
            "valid": "TinyStoriesV2-GPT4-valid.txt",
        },
        "separator": "<|endoftext|>",
    },
    # FineWeb-Edu is a parquet dataset; reading parquet needs pyarrow, which is
    # optional (pip install pyarrow). Licence: ODC-By 1.0.
    "fineweb-edu-10bt": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "license": "ODC-By-1.0",
        "paper": "Penedo et al. 2024, arXiv:2406.17557",
        # two shards of the 10BT sample: one for training, one held out for validation
        "files": {"train": "sample/10BT/000_00000.parquet", "valid": "sample/10BT/001_00000.parquet"},
        "separator": None,
    },
}


def url_for(repo: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/main/{path}"


def fetch(url: str, dest: str, max_bytes: int | None = None, chunk: int = 1 << 20) -> dict:
    """Stream ``url`` to ``dest``; optionally only the first ``max_bytes`` bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "tinylm-downloader/0.1"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    h = hashlib.sha256()
    n = 0
    t0 = time.time()
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
            if n % (64 << 20) < chunk:
                print(f"  {os.path.basename(dest)}: {n / 1e6:.0f} MB / "
                      f"{int(total) / 1e6 if total else float('nan'):.0f} MB", flush=True)
    os.replace(tmp, dest)
    return {"url": url, "bytes": n, "sha256": h.hexdigest(), "range": [0, n - 1] if max_bytes else None,
            "seconds": round(time.time() - t0, 1)}


def parquet_to_text(src: str, dest: str, column: str = "text", sep: str = "<|endoftext|>") -> int:
    """Write the ``column`` of a parquet file as separator-delimited text (needs pyarrow)."""
    import pyarrow.parquet as pq  # optional dependency, only for parquet corpora
    n = 0
    with open(dest, "w", encoding="utf-8", newline="") as f:
        for batch in pq.ParquetFile(src).iter_batches(columns=[column], batch_size=4096):
            for t in batch.column(0).to_pylist():
                f.write(t.replace(sep, " ") + "\n" + sep + "\n")
                n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", choices=sorted(CORPORA))
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-mb", type=float, default=1000.0,
                    help="take at most this many MB of each text file (prefix via HTTP Range); "
                         "0 = whole file (required for parquet, which is then converted to text)")
    ap.add_argument("--splits", nargs="*", default=None)
    a = ap.parse_args(argv)
    spec = CORPORA[a.corpus]
    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "SOURCES.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for split, path in spec["files"].items():
        if a.splits and split not in a.splits:
            continue
        dest = os.path.join(a.out, f"{a.corpus}-{split}{os.path.splitext(path)[1]}")
        url = url_for(spec["repo"], path)
        print(f"downloading {url} -> {dest}", flush=True)
        rec = fetch(url, dest, int(a.max_mb * 1e6) if a.max_mb > 0 else None)
        rec.update(license=spec["license"], paper=spec["paper"], repo=spec["repo"], file=path)
        if dest.endswith(".parquet"):
            if a.max_mb > 0:
                raise SystemExit("parquet files cannot be truncated; use --max-mb 0")
            n = parquet_to_text(dest, dest[:-len(".parquet")] + ".txt")
            rec["documents"] = n
        meta[f"{a.corpus}/{split}"] = rec
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"  done: {rec['bytes'] / 1e6:.1f} MB sha256={rec['sha256'][:16]}...", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
