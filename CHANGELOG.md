# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-26

First public release: a small LLM trained from scratch, end to end.

### Added
- Dataset download over HTTP range requests with a provenance record (URL, bytes, SHA-256, licence)
- Cleaning: NFKC + typographic folding, length / alphabetic / English / repetition heuristics
- Exact (BLAKE2b) and near-duplicate (MinHash + LSH banding + union-find) deduplication
- Hand-written Qwen2 pre-tokeniser (checked against the real regex) and an incremental byte-level BPE trainer
- Sharded uint16 token files with a deterministic, rank-aware random-window sampler
- Qwen2-architecture decoder (RMSNorm, NeoX RoPE, GQA, SwiGLU, QKV bias, tied embeddings) from tensor primitives
- Hand-written AdamW, cosine schedule with warm-up, global-norm clipping and dynamic fp16 loss scaling
- Training loop with gradient accumulation, exact checkpoint/resume, held-out evaluation and a `grad_sync` hook for data parallelism (tested with 2-process gloo)
- Evaluation harness: perplexity, a 4-way story-cloze task and BLiMP-style grammar minimal pairs
- GGUF export (F32/F16) that tinyinfer runs unmodified; logits parity tests against tinyinfer's NumPy reference and C engine
- Trained models (tl-25m target, tl-3m draft) with their real loss curves, evaluation scores and tinyinfer speculative-decoding speed

[0.1.0]: https://github.com/coopersimpson101-debug/tinylm/releases/tag/v0.1.0
