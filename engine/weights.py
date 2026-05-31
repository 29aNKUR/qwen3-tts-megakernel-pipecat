"""
Weight loading for Qwen3-TTS-12Hz-0.6B-Base.

Verified against the actual model structure:

TALKER (28 layers, backbone identical to Qwen3-0.6B):
  talker.model.layers.{i}.input_layernorm.weight        (1024,)
  talker.model.layers.{i}.self_attn.q_proj.weight       (2048, 1024)
  talker.model.layers.{i}.self_attn.k_proj.weight       (1024, 1024)
  talker.model.layers.{i}.self_attn.v_proj.weight       (1024, 1024)
  talker.model.layers.{i}.self_attn.q_norm.weight       (128,)
  talker.model.layers.{i}.self_attn.k_norm.weight       (128,)
  talker.model.layers.{i}.self_attn.o_proj.weight       (1024, 2048)
  talker.model.layers.{i}.post_attention_layernorm.weight (1024,)
  talker.model.layers.{i}.mlp.gate_proj.weight          (3072, 1024)
  talker.model.layers.{i}.mlp.up_proj.weight            (3072, 1024)
  talker.model.layers.{i}.mlp.down_proj.weight          (1024, 3072)
  talker.model.codec_embedding.weight                   (3072, 1024)  audio code input
  talker.model.text_embedding.weight                    (151936, 2048) text input
  talker.model.norm.weight                              (1024,)
  talker.codec_head.weight                              (3072, 1024)  output head

CODE PREDICTOR (5 layers, same backbone shape):
  talker.code_predictor.model.layers.{i}.*              (same structure)
  talker.code_predictor.model.codec_embedding.{g}.weight (2048, 1024) per group g=0..14
  talker.code_predictor.lm_head.{g}.weight              (2048, 1024)  per group g=0..14

So the talker emits codebook 0 (vocab 3072). The code predictor emits
codebooks 1..15 (vocab 2048 each), with a separate embedding + head per group.
Both backbones are byte-for-byte compatible with the megakernel layer format.
"""

import struct
import glob
import os
import torch
from safetensors.torch import load_file

# --- Talker constants ---
TALKER_NUM_LAYERS = 28
TALKER_VOCAB_SIZE = 3072        # codec_head output

# --- Code predictor constants ---
PREDICTOR_NUM_LAYERS = 5
PREDICTOR_VOCAB_SIZE = 2048      # each lm_head output
NUM_PREDICTOR_GROUPS = 15        # codebooks 1..15

# --- Shared architecture (talker and predictor identical) ---
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048

# Total codebooks per frame: 1 talker + 15 predictor
NUM_CODEBOOKS = 16

# RoPE frequency. Qwen3 uses 1,000,000 for the 0.6B family.
ROPE_THETA = 1_000_000.0


def _build_rope_tables(max_seq_len=MAX_SEQ_LEN, theta=ROPE_THETA):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def _extract_layer_weights(state, prefix, num_layers):
    """Pull the 11 tensors per layer in the exact order the kernel expects."""
    layer_weights = []
    for i in range(num_layers):
        p = f"{prefix}{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])
    return layer_weights


def pack_layer_weights(layer_weights, num_layers):
    """Pack into the LDGLayerWeights struct blob (11 x 8-byte pointers per layer)."""
    ptr_size = 8
    n_ptrs = 11
    buf = bytearray(num_layers * n_ptrs * ptr_size)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def load_tts_weights(model_dir="./qwen3-tts-model", verbose=True):
    """
    Load Qwen3-TTS-12Hz-0.6B-Base weights into GPU tensors.

    model_dir is the local directory the model was downloaded to.
    """
    if verbose:
        print(f"Loading weights from {model_dir}...")

    shard_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    state = {}
    for shard in shard_files:
        state.update(load_file(shard, device="cuda"))

    # Convert everything to bfloat16 (model may ship some fp32)
    for k in list(state.keys()):
        if state[k].dtype == torch.float32:
            state[k] = state[k].to(torch.bfloat16)

    if verbose:
        print(f"Loaded {len(state)} tensors. Building RoPE tables...")

    cos_table, sin_table = _build_rope_tables()

    # --- Talker backbone ---
    talker_layer_weights = _extract_layer_weights(
        state, "talker.model.layers.", TALKER_NUM_LAYERS
    )
    talker_codec_embed = state["talker.model.codec_embedding.weight"].contiguous()  # (3072,1024)
    talker_text_embed = state["talker.model.text_embedding.weight"].contiguous()    # (151936,2048)
    talker_norm = state["talker.model.norm.weight"].contiguous()
    talker_head = state["talker.codec_head.weight"].contiguous()                    # (3072,1024)

    # Text projection (projects 2048-dim text embedding down to 1024 hidden)
    text_proj_fc1_w = state["talker.text_projection.linear_fc1.weight"].contiguous()
    text_proj_fc1_b = state["talker.text_projection.linear_fc1.bias"].contiguous()
    text_proj_fc2_w = state["talker.text_projection.linear_fc2.weight"].contiguous()
    text_proj_fc2_b = state["talker.text_projection.linear_fc2.bias"].contiguous()

    # --- Code predictor backbone ---
    pred_layer_weights = _extract_layer_weights(
        state, "talker.code_predictor.model.layers.", PREDICTOR_NUM_LAYERS
    )
    pred_norm = state["talker.code_predictor.model.norm.weight"].contiguous()

    # 15 per-group embeddings and 15 per-group heads
    pred_codec_embeds = [
        state[f"talker.code_predictor.model.codec_embedding.{g}.weight"].contiguous()
        for g in range(NUM_PREDICTOR_GROUPS)
    ]
    pred_heads = [
        state[f"talker.code_predictor.lm_head.{g}.weight"].contiguous()
        for g in range(NUM_PREDICTOR_GROUPS)
    ]

    # Pack layer weights into kernel struct format
    talker_packed = pack_layer_weights(talker_layer_weights, TALKER_NUM_LAYERS)
    pred_packed = pack_layer_weights(pred_layer_weights, PREDICTOR_NUM_LAYERS)

    weights = {
        # Talker
        "talker_codec_embed": talker_codec_embed,
        "talker_text_embed": talker_text_embed,
        "talker_layer_weights": talker_layer_weights,
        "talker_layer_weights_packed": talker_packed,
        "talker_norm": talker_norm,
        "talker_head": talker_head,
        "text_proj_fc1_w": text_proj_fc1_w,
        "text_proj_fc1_b": text_proj_fc1_b,
        "text_proj_fc2_w": text_proj_fc2_w,
        "text_proj_fc2_b": text_proj_fc2_b,

        # Code predictor
        "pred_layer_weights": pred_layer_weights,
        "pred_layer_weights_packed": pred_packed,
        "pred_norm": pred_norm,
        "pred_codec_embeds": pred_codec_embeds,   # list of 15
        "pred_heads": pred_heads,                 # list of 15

        # Shared
        "cos_table": cos_table,
        "sin_table": sin_table,

        "model_dir": model_dir,
        "_state": state,  # keep alive so tensors are not freed
    }

    if verbose:
        print("Talker: 28 layers, vocab 3072")
        print("Code predictor: 5 layers, 15 groups, vocab 2048 each")
        print("All weights packed.")

    return weights
