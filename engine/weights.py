"""
Weight loading for Qwen3-TTS talker decoder and code predictor.

The Qwen3-TTS talker decoder is architecturally identical to Qwen3-0.6B:
  - Same hidden size: 1024
  - Same 28 layers
  - Same 16 query heads, 8 KV heads, head dim 128
  - Different vocab: 3072 audio codes (not 151936 text tokens)
  - Different RoPE frequency: 1,000,000 (not 10,000)
  - Untied output embeddings (separate lm_head weight)

The code predictor is the same architecture but only 5 layers.
We run both through the same compiled megakernel by swapping weights
and passing num_layers=5 for the code predictor.
"""

import struct
import torch

# --- Talker decoder constants ---
TALKER_NUM_LAYERS = 28
TALKER_VOCAB_SIZE = 3072

# --- Code predictor constants ---
PREDICTOR_NUM_LAYERS = 5

# --- Shared architecture constants (same for both) ---
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_Q_HEADS = 16
Q_SIZE = NUM_Q_HEADS * HEAD_DIM   # 2048
KV_SIZE = NUM_KV_HEADS * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048

# RoPE frequency for TTS (different from text model's 10000)
ROPE_THETA = 1_000_000.0

# Number of codebook groups Qwen3-TTS uses
NUM_CODEBOOKS = 16  # 1 from talker + 15 from code predictor


def _build_rope_tables(max_seq_len: int = MAX_SEQ_LEN, theta: float = ROPE_THETA):
    """Build cos/sin RoPE tables for the given theta frequency."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def _extract_layer_weights(state: dict, prefix: str, num_layers: int) -> list:
    """
    Extract the 11 weight tensors per layer that the megakernel expects.
    Returns a flat list: [layer0_w0, layer0_w1, ..., layer0_w10, layer1_w0, ...]
    """
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


def pack_layer_weights(layer_weights: list, num_layers: int) -> torch.Tensor:
    """
    Pack the flat layer weight list into a GPU blob of LDGLayerWeights structs.
    This is the same packing format the original megakernel expects.
    Each struct is 11 x 8-byte pointers = 88 bytes.
    """
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(num_layers * struct_bytes)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def load_tts_weights(model_name: str = "Qwen/Qwen3-TTS", verbose: bool = True):
    """
    Load Qwen3-TTS weights from HuggingFace.

    Returns a dict with everything both the talker and code predictor need.
    We load the full model once, extract weights for both sub-models, then
    delete the HuggingFace model to free CPU memory.
    """
    if verbose:
        print(f"Loading {model_name} from HuggingFace...")

    # We load the modeling file directly since Qwen3-TTS uses a custom
    # generation script (modeling_qwen3_tts.py) not the standard AutoModel
    from transformers import AutoProcessor
    import os

    # Download model files
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(model_name)

    if verbose:
        print(f"Model downloaded to: {model_path}")
        print("Loading weights into GPU tensors...")

    # Load safetensors directly for efficiency
    from safetensors.torch import load_file
    import glob

    shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    state = {}
    for shard in shard_files:
        state.update(load_file(shard, device="cuda"))

    if verbose:
        print(f"Loaded {len(state)} weight tensors")

    # Build RoPE tables (TTS uses theta=1,000,000 not 10,000)
    cos_table, sin_table = _build_rope_tables()

    # --- Talker decoder weights ---
    # The talker uses the same 28-layer Qwen3 backbone
    talker_layer_weights = _extract_layer_weights(
        state, "talker.model.layers.", TALKER_NUM_LAYERS
    )
    talker_embed = state["talker.model.embed_tokens.weight"].contiguous()
    talker_final_norm = state["talker.model.norm.weight"].contiguous()
    # Qwen3-TTS has UNTIED lm_head (separate from embed_tokens)
    talker_lm_head = state["talker.lm_head.weight"].contiguous()

    # --- Code predictor weights ---
    # Same architecture, 5 layers instead of 28
    predictor_layer_weights = _extract_layer_weights(
        state, "code_predictor.model.layers.", PREDICTOR_NUM_LAYERS
    )
    predictor_embed = state["code_predictor.model.embed_tokens.weight"].contiguous()
    predictor_final_norm = state["code_predictor.model.norm.weight"].contiguous()
    predictor_lm_head = state["code_predictor.lm_head.weight"].contiguous()

    # --- Vocoder weights (kept as HF model for simplicity) ---
    # The vocoder converts audio codes back to waveforms
    # We keep this in PyTorch since it is not the bottleneck
    vocoder_path = os.path.join(model_path, "vocoder")

    # Pre-pack layer weights into GPU structs the kernel expects
    talker_packed = pack_layer_weights(talker_layer_weights, TALKER_NUM_LAYERS)
    predictor_packed = pack_layer_weights(predictor_layer_weights, PREDICTOR_NUM_LAYERS)

    weights = {
        # Talker
        "talker_embed": talker_embed,
        "talker_layer_weights": talker_layer_weights,
        "talker_layer_weights_packed": talker_packed,
        "talker_final_norm": talker_final_norm,
        "talker_lm_head": talker_lm_head,

        # Code predictor
        "predictor_embed": predictor_embed,
        "predictor_layer_weights": predictor_layer_weights,
        "predictor_layer_weights_packed": predictor_packed,
        "predictor_final_norm": predictor_final_norm,
        "predictor_lm_head": predictor_lm_head,

        # Shared RoPE tables
        "cos_table": cos_table,
        "sin_table": sin_table,

        # Paths for vocoder
        "model_path": model_path,
        "vocoder_path": vocoder_path,
    }

    if verbose:
        print("All weights loaded and packed successfully.")

    return weights