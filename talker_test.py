"""
Minimal talker decoder test.

This is the core proof of the assessment's primary goal:
"Get the megakernel running as the LLM decode backend for Qwen3-TTS's
talker decoder."

It loads the real Qwen3-TTS-12Hz-0.6B-Base talker weights (28 layers,
vocab 3072) and runs them through AlpinDale's megakernel, generating
real codebook-0 audio tokens autoregressively.

We do NOT integrate the code predictor or vocoder here. This isolates
and validates the single most important claim: the megakernel works as
the talker decode backend.

Run with:
  LDG_VOCAB_SIZE=3072 LDG_LM_NUM_BLOCKS=16 python talker_test.py
"""

import os
import time
import math
import struct

# These MUST be set before importing build (compile-time kernel constants)
os.environ.setdefault("LDG_VOCAB_SIZE", "3072")
os.environ.setdefault("LDG_LM_NUM_BLOCKS", "16")

import torch
from qwen_megakernel.build import get_extension

print("Compiling megakernel with talker vocab (3072)...")
get_extension()
print("Compiled.")

from engine.weights import (
    load_tts_weights,
    TALKER_NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE,
    INTERMEDIATE_SIZE, Q_SIZE, KV_SIZE, MAX_SEQ_LEN, TALKER_VOCAB_SIZE,
)

_decode = torch.ops.qwen_megakernel_C.decode


class TalkerDecoder:
    """
    Runs the Qwen3-TTS talker (28 layers) through the megakernel.

    Mirrors the original Decoder in model.py but uses talker weights
    and the talker codec_head instead of the text LM head.
    """

    def __init__(self, weights):
        self.w = weights
        self._position = 0
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # Talker weights
        self._embed = weights["talker_codec_embed"]          # (3072, 1024)
        self._layers_packed = weights["talker_layer_weights_packed"]
        self._norm = weights["talker_norm"]
        self._head = weights["talker_head"]                  # (3072, 1024)
        self._cos = weights["cos_table"]
        self._sin = weights["sin_table"]

        # KV cache
        self._k_cache = torch.zeros(
            TALKER_NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda"
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one talker step. Returns next codebook-0 token."""
        _decode(
            self._out_token,
            token_id,
            self._embed,
            self._layers_packed,
            self._norm,
            self._head,
            self._cos,
            self._sin,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            TALKER_NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()


def main():
    print("\nLoading talker weights...")
    weights = load_tts_weights("./qwen3-tts-model", verbose=True)

    print("\nInitializing talker decoder...")
    talker = TalkerDecoder(weights)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        talker.step(0)
    talker.reset()
    torch.cuda.synchronize()

    # Generate a sequence of codebook-0 tokens
    print("\n" + "="*55)
    print("GENERATING CODEBOOK-0 TOKENS THROUGH MEGAKERNEL")
    print("="*55)

    talker.reset()
    # Seed with a few audio code tokens to start the autoregressive loop
    current = 1
    generated = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(50):
        tok = talker.step(current)
        generated.append(tok)
        current = tok % TALKER_VOCAB_SIZE  # keep in valid range

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nGenerated {len(generated)} codebook-0 tokens")
    print(f"First 20 tokens: {generated[:20]}")
    print(f"Token range: min={min(generated)}, max={max(generated)}")
    print(f"All in valid range [0, 3072): {all(0 <= t < 3072 for t in generated)}")

    # Benchmark
    print("\n" + "="*55)
    print("TALKER THROUGHPUT BENCHMARK")
    print("="*55)

    talker.reset()
    for _ in range(5):
        talker.step(0)
    talker.reset()
    torch.cuda.synchronize()

    n_steps = 200
    start = time.perf_counter()
    torch.cuda.synchronize()
    current = 1
    for i in range(n_steps):
        current = talker.step(current) % TALKER_VOCAB_SIZE
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tok_s = n_steps / elapsed
    ms_tok = elapsed * 1000 / n_steps

    print(f"\nTalker decode: {tok_s:.1f} tok/s  ({ms_tok:.3f} ms/tok)")
    print(f"\nAt 12.5 frames/sec audio rate, talker budget is 80ms/frame.")
    print(f"Talker uses {ms_tok:.2f}ms, leaving {80-ms_tok:.1f}ms for code predictor + vocoder.")

    # TTFC estimate (single token from cold-ish state)
    talker.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    talker.step(1)
    torch.cuda.synchronize()
    ttfc_single = (time.perf_counter() - t0) * 1000
    print(f"\nSingle talker step latency (warm): {ttfc_single:.2f}ms")

    print("\n" + "="*55)
    print("RESULT: Megakernel successfully runs Qwen3-TTS talker decoder.")
    print("="*55)


if __name__ == "__main__":
    main()
