#!/usr/bin/env python3
"""Convert a HuggingFace LLaMA-family model to mini_infer format.

Usage:
    # From a local snapshot (already downloaded):
    python tools/convert_hf.py  path/to/model  ./models/smollm

    # Download first, then convert:
    python tools/convert_hf.py  HuggingFaceTB/SmolLM2-135M  ./models/smollm  --download

Creates:
    <output_dir>/model.bin       — weights  (v2 format)
    <output_dir>/tokenizer.bin   — BPE tokenizer

Requirements:
    pip install safetensors numpy huggingface_hub
"""

import argparse, json, os, struct, sys
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  GPT-2 byte encoder — maps each byte to a printable unicode char
# ════════════════════════════════════════════════════════════════════

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def token_str_to_bytes(s):
    """Convert a GPT-2/BPE token string to raw bytes."""
    try:
        return bytes([BYTE_DECODER[c] for c in s])
    except KeyError:
        return s.encode("utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════
#  Model conversion
# ════════════════════════════════════════════════════════════════════

def convert_model(model_dir, output_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"ERROR: {cfg_path} not found")

    with open(cfg_path) as f:
        cfg = json.load(f)

    d_model     = cfg["hidden_size"]
    n_heads     = cfg["num_attention_heads"]
    n_kv_heads  = cfg.get("num_key_value_heads", n_heads)
    d_head      = d_model // n_heads
    d_ff        = cfg["intermediate_size"]
    n_layers    = cfg["num_hidden_layers"]
    vocab_size  = cfg["vocab_size"]
    max_seq_len = cfg.get("max_position_embeddings", 2048)
    norm_eps    = cfg.get("rms_norm_eps", 1e-5)
    rope_theta  = cfg.get("rope_theta", 10000.0)
    tie_emb     = cfg.get("tie_word_embeddings", False)

    print(f"Model config:")
    print(f"  d_model={d_model}  n_heads={n_heads}  n_kv_heads={n_kv_heads}  d_head={d_head}")
    print(f"  d_ff={d_ff}  n_layers={n_layers}  vocab_size={vocab_size}")
    print(f"  max_seq_len={max_seq_len}  rope_theta={rope_theta}  tie={tie_emb}")

    # Load safetensors — use torch for bf16 support, fall back to numpy
    weights = {}
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith(".safetensors"):
            print(f"  loading {fname}")
            try:
                from safetensors.torch import load_file as load_torch
                import torch
                w = load_torch(os.path.join(model_dir, fname))
                w = {k: v.float().numpy() for k, v in w.items()}
            except ImportError:
                from safetensors.numpy import load_file as load_np
                w = load_np(os.path.join(model_dir, fname))
            weights.update(w)
    if not weights:
        sys.exit("ERROR: no .safetensors files found")

    print(f"  loaded {len(weights)} tensors")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.bin")

    with open(out_path, "wb") as f:
        # ── Header (v2 format) ──
        f.write(struct.pack("<I", 0x4D494E49))   # magic "MINI" as LE u32
        f.write(struct.pack("<i", 2))             # version
        f.write(struct.pack("<i", d_model))
        f.write(struct.pack("<i", n_heads))
        f.write(struct.pack("<i", n_kv_heads))
        f.write(struct.pack("<i", d_head))
        f.write(struct.pack("<i", d_ff))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", max_seq_len))
        f.write(struct.pack("<f", norm_eps))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<i", 1))            # ffn_type = SwiGLU

        def w(name, expected_shape=None):
            t = weights[name].astype(np.float32)
            if expected_shape:
                eshape = tuple(expected_shape)
                assert t.shape == eshape, f"{name}: expected {eshape}, got {t.shape}"
            f.write(t.tobytes())
            print(f"    {name:50s} {str(t.shape):20s} {t.nbytes/1e6:.1f} MB")
            return t

        # ── Weights ──
        print("Writing weights:")
        tok_emb = w("model.embed_tokens.weight", [vocab_size, d_model])

        for i in range(n_layers):
            p = f"model.layers.{i}"
            w(f"{p}.self_attn.q_proj.weight", [n_heads * d_head, d_model])
            w(f"{p}.self_attn.k_proj.weight", [n_kv_heads * d_head, d_model])
            w(f"{p}.self_attn.v_proj.weight", [n_kv_heads * d_head, d_model])
            w(f"{p}.self_attn.o_proj.weight", [d_model, n_heads * d_head])
            w(f"{p}.mlp.gate_proj.weight",    [d_ff, d_model])
            w(f"{p}.mlp.down_proj.weight",    [d_model, d_ff])
            w(f"{p}.mlp.up_proj.weight",      [d_ff, d_model])
            w(f"{p}.input_layernorm.weight",  [d_model])
            w(f"{p}.post_attention_layernorm.weight", [d_model])

        w("model.norm.weight", [d_model])

        if tie_emb or "lm_head.weight" not in weights:
            print(f"    {'(tied lm_head = embed_tokens)':50s}")
            f.write(tok_emb.tobytes())
        else:
            w("lm_head.weight", [vocab_size, d_model])

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════
#  Tokenizer conversion
# ════════════════════════════════════════════════════════════════════

def convert_tokenizer(model_dir, output_dir):
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        sys.exit(f"ERROR: {tok_path} not found")

    with open(tok_path, encoding="utf-8") as f:
        data = json.load(f)

    model_data = data.get("model", {})
    vocab = model_data.get("vocab", {})
    merges = model_data.get("merges", [])

    # Determine vocab_size from the largest id
    vocab_size = max(vocab.values()) + 1 if vocab else 0
    print(f"\nTokenizer: vocab_size={vocab_size}  merges={len(merges)}")

    # Special tokens
    added = {t["content"]: t["id"] for t in data.get("added_tokens", [])}
    bos = added.get("<s>", added.get("<|begin_of_text|>",
          added.get("<|endoftext|>", 0)))
    eos = added.get("</s>", added.get("<|end_of_text|>",
          added.get("<|endoftext|>", 0)))
    print(f"  bos_token={bos}  eos_token={eos}")

    # Convert each token to raw bytes
    token_bytes = [b""] * vocab_size
    for tok_str, tok_id in vocab.items():
        if tok_id < vocab_size:
            token_bytes[tok_id] = token_str_to_bytes(tok_str)

    # Also include added tokens
    for t in data.get("added_tokens", []):
        tid = t["id"]
        if tid < vocab_size:
            token_bytes[tid] = t["content"].encode("utf-8", errors="replace")

    # Build byte_to_token[256]
    byte_to_token = [0] * 256
    for byte_val in range(256):
        char = BYTE_ENCODER[byte_val]
        tid = vocab.get(char, 0)
        byte_to_token[byte_val] = tid

    # Process merges
    merge_entries = []
    for merge_str in merges:
        parts = merge_str.split(" ", 1)
        if len(parts) != 2:
            continue
        a_str, b_str = parts
        result_str = a_str + b_str
        a_id = vocab.get(a_str)
        b_id = vocab.get(b_str)
        result_id = vocab.get(result_str)
        if a_id is not None and b_id is not None and result_id is not None:
            merge_entries.append((a_id, b_id, result_id))

    print(f"  valid merges: {len(merge_entries)}")

    # Write tokenizer.bin
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tokenizer.bin")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0x4D49544B))   # magic "MITK" as LE u32
        f.write(struct.pack("<i", 1))             # version
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", bos))
        f.write(struct.pack("<i", eos))
        f.write(struct.pack("<i", len(merge_entries)))

        # byte_to_token[256]
        for bt in byte_to_token:
            f.write(struct.pack("<i", bt))

        # vocab entries
        for tid in range(vocab_size):
            raw = token_bytes[tid]
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)

        # merge entries
        for a_id, b_id, result_id in merge_entries:
            f.write(struct.pack("<iii", a_id, b_id, result_id))

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace LLaMA model to mini_infer format")
    parser.add_argument("model_path",
        help="Local model directory or HuggingFace model ID")
    parser.add_argument("output_dir",
        help="Output directory for model.bin and tokenizer.bin")
    parser.add_argument("--download", action="store_true",
        help="Download from HuggingFace Hub first")
    args = parser.parse_args()

    model_dir = args.model_path
    if args.download:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.model_path} ...")
        model_dir = snapshot_download(args.model_path)
        print(f"Downloaded to {model_dir}\n")

    convert_model(model_dir, args.output_dir)
    convert_tokenizer(model_dir, args.output_dir)
    print("\nDone!  Next:")
    print(f"  ./examples/generate_real {args.output_dir}")
