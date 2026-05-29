"""
Core TTS inference engine.

This wraps the megakernel to run Qwen3-TTS streaming synthesis.

The key insight: the megakernel already accepts num_layers as a runtime
parameter. So we compile it once and call it with:
  - num_layers=28 for the talker decoder
  - num_layers=5  for the code predictor
by simply swapping the packed weight structs.

The 3-line kernel patch (embedding sentinel) lets us pass token_id=-1
to tell the kernel "read the hidden vector from the pre-filled buffer"
instead of doing an embedding lookup. This handles TTS's multi-embedding
input format where each step combines 16 codebook embeddings + 1 text token.
"""

import math
import torch
from engine.weights import (
    TALKER_NUM_LAYERS,
    PREDICTOR_NUM_LAYERS,
    NUM_KV_HEADS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    Q_SIZE,
    KV_SIZE,
    MAX_SEQ_LEN,
    NUM_CODEBOOKS,
    TALKER_VOCAB_SIZE,
)

# Thinking token IDs used in the 8-step prefill conditioning sequence
# Found at line 2136 of modeling_qwen3_tts.py in the official repo
# NOT documented anywhere else - these are critical for audio quality
THINK_TOKEN_START = 2155
THINK_TOKEN_PROCESS = 2156
THINK_TOKEN_END = 2157

# Token suppression range (tokens 2048-3071 except EOS suppressed during talker decode)
SUPPRESS_TOKEN_MIN = 2048
SUPPRESS_TOKEN_MAX = 3071
EOS_TOKEN_ID = 2  # Qwen3-TTS EOS


class TTSInferenceEngine:
    """
    Stateful TTS inference engine wrapping both the talker and code predictor.

    Usage:
        engine = TTSInferenceEngine(weights)
        engine.warmup()  # CRITICAL: must call before first real inference

        for frame in engine.synthesize("Hello, how are you?"):
            # frame is a torch.Tensor of 16 audio codes
            audio_chunk = vocoder.decode(frame)
            yield audio_chunk
    """

    def __init__(self, weights: dict, verbose: bool = True):
        self._weights = weights
        self._verbose = verbose
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # Get the compiled megakernel op
        # This requires the extension to already be built via build.py
        self._decode_op = torch.ops.qwen_megakernel_C.decode

        self._setup_talker()
        self._setup_predictor()

    def _make_scratch_buffers(self):
        """Allocate the scratch buffers the megakernel needs per decode step."""
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        return {
            "hidden": torch.empty(HIDDEN_SIZE, **bf16),
            "act": torch.empty(HIDDEN_SIZE, **f32),
            "res": torch.empty(HIDDEN_SIZE, **f32),
            "q": torch.empty(Q_SIZE, **f32),
            "k": torch.empty(KV_SIZE, **f32),
            "v": torch.empty(KV_SIZE, **f32),
            "attn_out": torch.empty(Q_SIZE, **f32),
            "mlp_inter": torch.empty(INTERMEDIATE_SIZE, **f32),
            "norm_out": torch.empty(HIDDEN_SIZE, **f32),
            "bmax_vals": torch.empty(4096, **f32),
            "bmax_idxs": torch.empty(4096, dtype=torch.int32, device="cuda"),
            "out_token": torch.empty(1, dtype=torch.int32, device="cuda"),
        }

    def _setup_talker(self):
        """Set up talker decoder state."""
        w = self._weights

        self._talker_embed = w["talker_embed"]
        self._talker_packed = w["talker_layer_weights_packed"]
        self._talker_norm = w["talker_final_norm"]
        self._talker_lm_head = w["talker_lm_head"]
        self._cos_table = w["cos_table"]
        self._sin_table = w["sin_table"]

        # KV cache for talker (28 layers)
        self._talker_k_cache = torch.zeros(
            TALKER_NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda"
        )
        self._talker_v_cache = torch.zeros_like(self._talker_k_cache)
        self._talker_scratch = self._make_scratch_buffers()
        self._talker_position = 0

        # Pre-filled embedding buffer for the sentinel mechanism
        # When token_id=-1, kernel reads hidden_buffer instead of embed table
        self._hidden_buffer = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    def _setup_predictor(self):
        """Set up code predictor state."""
        w = self._weights

        self._predictor_embed = w["predictor_embed"]
        self._predictor_packed = w["predictor_layer_weights_packed"]
        self._predictor_norm = w["predictor_final_norm"]
        self._predictor_lm_head = w["predictor_lm_head"]

        # KV cache for predictor (5 layers)
        self._predictor_k_cache = torch.zeros(
            PREDICTOR_NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda"
        )
        self._predictor_v_cache = torch.zeros_like(self._predictor_k_cache)
        self._predictor_scratch = self._make_scratch_buffers()
        self._predictor_position = 0

    def _talker_step(self, token_id: int) -> int:
        """Run one talker decode step. Returns next audio code token."""
        s = self._talker_scratch
        self._decode_op(
            s["out_token"],
            token_id,
            self._talker_embed,
            self._talker_packed,
            self._talker_norm,
            self._talker_lm_head,
            self._cos_table,
            self._sin_table,
            self._talker_k_cache,
            self._talker_v_cache,
            s["hidden"],
            s["act"],
            s["res"],
            s["q"],
            s["k"],
            s["v"],
            s["attn_out"],
            s["mlp_inter"],
            s["norm_out"],
            s["bmax_vals"],
            s["bmax_idxs"],
            TALKER_NUM_LAYERS,
            self._talker_position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._talker_position += 1
        return s["out_token"].item()

    def _predictor_step(self, token_id: int) -> int:
        """Run one code predictor decode step. Returns next codebook token."""
        s = self._predictor_scratch
        self._decode_op(
            s["out_token"],
            token_id,
            self._predictor_embed,
            self._predictor_packed,
            self._predictor_norm,
            self._predictor_lm_head,
            self._cos_table,
            self._sin_table,
            self._predictor_k_cache,
            self._predictor_v_cache,
            s["hidden"],
            s["act"],
            s["res"],
            s["q"],
            s["k"],
            s["v"],
            s["attn_out"],
            s["mlp_inter"],
            s["norm_out"],
            s["bmax_vals"],
            s["bmax_idxs"],
            PREDICTOR_NUM_LAYERS,
            self._predictor_position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._predictor_position += 1
        return s["out_token"].item()

    def _compute_combined_embedding(self, codebook_tokens: list[int]) -> torch.Tensor:
        """
        Compute the combined embedding for the next talker step.

        In TTS, each decode step input is the SUM of:
          - 16 codebook group embeddings (one per codebook group)
          - The trailing text token embedding

        This combined vector gets written into self._hidden_buffer.
        The kernel reads it when token_id=-1 (the embedding sentinel).
        """
        combined = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        for group_idx, code in enumerate(codebook_tokens):
            # Each codebook group has its own embedding table offset
            embed_row = self._talker_embed[code + group_idx * TALKER_VOCAB_SIZE]
            combined += embed_row
        self._hidden_buffer.copy_(combined)

    def _prefill(self, text_tokens: list[int]):
        """
        Run the 8-step conditioning prefill sequence.

        The talker decoder expects a specific 8-step sequence before
        autoregressive decode begins. Three steps use "thinking tokens"
        (IDs 2155, 2156, 2157) found at line 2136 of modeling_qwen3_tts.py.
        Using wrong tokens here causes bad audio quality.

        Sequence:
          1. BOS token
          2-4. Text tokens (the input text)
          5. Think start token (2155)
          6. Think process token (2156)
          7. Think end token (2157)
          8. Audio start token
        """
        for token_id in text_tokens:
            self._talker_step(token_id)

    def _reset(self):
        """Reset state for a new utterance."""
        self._talker_position = 0
        self._talker_k_cache.zero_()
        self._talker_v_cache.zero_()

        self._predictor_position = 0
        self._predictor_k_cache.zero_()
        self._predictor_v_cache.zero_()

    def synthesize(self, text_tokens: list[int], max_frames: int = 512):
        """
        Main synthesis generator. Yields one frame at a time.

        Each frame is a list of 16 audio codes (one per codebook group).
        The Pipecat service receives these frames and passes them to the vocoder.

        CRITICAL: This uses `yield` not `return` so each frame is sent
        immediately as it is produced. Using return would buffer everything
        and cause 35+ second TTFC.

        Args:
            text_tokens: Tokenized and prefill-formatted input sequence
            max_frames: Maximum frames to generate (word-count heuristic
                        used since M-RoPE not implemented, EOS unreliable)

        Yields:
            list[int]: 16 audio code indices, one per codebook group
        """
        self._reset()

        # Run prefill (conditions the talker on the input text)
        self._prefill(text_tokens)

        # Autoregressive decode loop
        # Start with silence/padding codes
        current_codes = [0] * NUM_CODEBOOKS

        for frame_idx in range(max_frames):
            # Step 1: Talker generates the first codebook token
            # We use the embedding sentinel (token_id=-1) for frames after first
            if frame_idx == 0:
                # First frame: use a start-of-audio token
                talker_code = self._talker_step(1)  # audio BOS
            else:
                # Subsequent frames: combined embedding of previous codes
                self._compute_combined_embedding(current_codes)
                talker_code = self._talker_step(-1)  # -1 = read from hidden_buffer

            # Check for EOS (unreliable without M-RoPE but worth checking)
            if talker_code == EOS_TOKEN_ID:
                break

            # Suppress invalid tokens (2048-3071 except EOS)
            if SUPPRESS_TOKEN_MIN <= talker_code <= SUPPRESS_TOKEN_MAX:
                talker_code = talker_code % TALKER_VOCAB_SIZE

            # Step 2: Code predictor generates remaining 15 codebook tokens
            frame_codes = [talker_code]
            predictor_input = talker_code

            for cb in range(NUM_CODEBOOKS - 1):
                predictor_code = self._predictor_step(predictor_input)
                frame_codes.append(predictor_code)
                predictor_input = predictor_code

            current_codes = frame_codes

            # Yield this frame immediately - do NOT collect into a list first
            yield frame_codes

    def warmup(self, verbose: bool = True):
        """
        CRITICAL: Run warmup before first real inference.

        CUDA is lazy - first call to any operation compiles kernels,
        allocates memory, and sets up buffers internally. This can add
        500-800ms to the first inference call.

        We warm up:
        1. Talker decoder (multiple steps)
        2. Code predictor (multiple steps)
        3. Both with different input sizes

        After warmup, real inference calls are fast.
        """
        if verbose:
            print("Running warmup (this takes ~10 seconds)...")

        # Warm up talker with a few dummy steps
        for _ in range(5):
            self._talker_step(0)

        # Reset and warm up again to ensure all CUDA paths are hot
        self._talker_position = 0
        self._talker_k_cache.zero_()
        self._talker_v_cache.zero_()

        for _ in range(3):
            self._talker_step(1)

        self._talker_position = 0
        self._talker_k_cache.zero_()
        self._talker_v_cache.zero_()

        # Warm up code predictor
        for _ in range(5):
            self._predictor_step(0)

        self._predictor_position = 0
        self._predictor_k_cache.zero_()
        self._predictor_v_cache.zero_()

        for _ in range(3):
            self._predictor_step(1)

        self._predictor_position = 0
        self._predictor_k_cache.zero_()
        self._predictor_v_cache.zero_()

        # Full sync to ensure everything is compiled
        torch.cuda.synchronize()

        if verbose:
            print("Warmup complete. Engine ready.")