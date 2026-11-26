"""Export a trained model + tokenizer to a Qwen2 GGUF that tinyinfer (and llama.cpp) can run.

    python -m tinylm.export runs/tl-25m/model.pt data/ts4k/tokenizer.json -o tl-25m-f32.gguf

Tensor names follow llama.cpp's Qwen2 convention::

    token_embd.weight                      (V, D)
    blk.{i}.attn_norm.weight               (D,)
    blk.{i}.attn_{q,k,v}.weight / .bias    (H*hd, D) / (H_kv*hd, D)
    blk.{i}.attn_output.weight             (D, H*hd)
    blk.{i}.ffn_norm.weight                (D,)
    blk.{i}.ffn_{gate,up}.weight           (F, D)
    blk.{i}.ffn_down.weight                (D, F)
    output_norm.weight                     (D,)
    output.weight                          only when embeddings are untied

The model uses the rotate-half RoPE layout, which is what llama.cpp / tinyinfer
apply for ``qwen2`` (LLAMA_ROPE_TYPE_NEOX), so Q/K weights need no permutation.
"""
import argparse

import torch

from .bpe import Tokenizer
from .gguf import GGUFWriter
from .model import Transformer


def export(model: Transformer, tok: Tokenizer, path: str, dtype: str = "f32", name: str = "tinylm"):
    c = model.cfg
    if tok.vocab_size != c.vocab_size:
        raise ValueError(f"tokenizer vocab {tok.vocab_size} != model vocab {c.vocab_size}")
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    w = GGUFWriter(path)
    w.string("general.architecture", "qwen2")
    w.string("general.name", name)
    w.u32("general.file_type", 1 if dtype == "f16" else 0)
    w.u32("qwen2.block_count", c.n_layers)
    w.u32("qwen2.context_length", c.max_seq_len)
    w.u32("qwen2.embedding_length", c.dim)
    w.u32("qwen2.feed_forward_length", c.ffn_dim)
    w.u32("qwen2.attention.head_count", c.n_heads)
    w.u32("qwen2.attention.head_count_kv", c.n_kv_heads)
    w.f32("qwen2.rope.freq_base", c.rope_theta)
    w.f32("qwen2.attention.layer_norm_rms_epsilon", c.rms_eps)
    w.string("tokenizer.ggml.model", "gpt2")
    w.string("tokenizer.ggml.pre", "qwen2")
    w.str_array("tokenizer.ggml.tokens", tok.tokens)
    w.i32_array("tokenizer.ggml.token_type", tok.types)
    w.str_array("tokenizer.ggml.merges", tok.merge_strings)
    w.u32("tokenizer.ggml.eos_token_id", tok.eos_id)
    w.u32("tokenizer.ggml.bos_token_id", tok.eos_id)
    w.u32("tokenizer.ggml.padding_token_id", tok.eos_id)
    w.boolean("tokenizer.ggml.add_bos_token", False)

    def mat(name, key):  # 2-D weights may be f16; 1-D vectors stay f32 (as llama.cpp does)
        w.tensor(name, sd[key], dtype)

    def vec(name, key):
        w.tensor(name, sd[key], "f32")

    mat("token_embd.weight", "tok_embd.weight")
    for i in range(c.n_layers):
        p, q = f"blk.{i}.", f"layers.{i}."
        vec(p + "attn_norm.weight", q + "attn_norm.weight")
        for s in "qkv":
            mat(p + f"attn_{s}.weight", q + f"attn.w{s}.weight")
            if c.qkv_bias:
                vec(p + f"attn_{s}.bias", q + f"attn.w{s}.bias")
        mat(p + "attn_output.weight", q + "attn.wo.weight")
        vec(p + "ffn_norm.weight", q + "ffn_norm.weight")
        mat(p + "ffn_gate.weight", q + "mlp.gate.weight")
        mat(p + "ffn_up.weight", q + "mlp.up.weight")
        mat(p + "ffn_down.weight", q + "mlp.down.weight")
    vec("output_norm.weight", "norm.weight")
    if not c.tied_embeddings:
        mat("output.weight", "output.weight")
    w.write()
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--name", default="tinylm")
    a = ap.parse_args(argv)
    from .train import load_model
    m = load_model(a.checkpoint)
    export(m, Tokenizer.load(a.tokenizer), a.out, a.dtype, a.name)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
