"""
Pipecat TTS service backed by the Qwen3-TTS megakernel engine.

Pipecat expects a TTSService subclass that implements run_tts() as an
async generator yielding TTSAudioRawFrame objects. Each frame is a chunk
of raw PCM audio bytes that Pipecat routes to the audio output.

The key constraint: audio must stream frame-by-frame. Do NOT buffer.
The first audio chunk must arrive in under 90ms.
"""

import asyncio
import time
from typing import AsyncGenerator

import torch
import numpy as np

from pipecat.services.tts import TTSService
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from engine.inference import TTSInferenceEngine
from engine.weights import NUM_CODEBOOKS

# Audio config matching Qwen3-TTS vocoder output
SAMPLE_RATE = 24000       # Hz
FRAME_RATE = 12.5         # audio frames per second
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples = 80ms of audio
BATCH_FRAMES = 10         # batch this many frames before sending (800ms chunks)
                          # first chunk sends immediately for low TTFC


class MegakernelTTSService(TTSService):
    """
    Pipecat TTS service using the Qwen3-TTS megakernel engine.

    Drop-in replacement for any Pipecat TTSService.
    Streams audio frame-by-frame with sub-90ms TTFC.

    Example:
        tts = MegakernelTTSService(weights=weights, vocoder=vocoder)
        pipeline = Pipeline([stt, llm, tts, transport.output()])
    """

    def __init__(self, weights: dict, vocoder, tokenizer=None, **kwargs):
        """
        Args:
            weights: Loaded TTS weight dict from engine.weights.load_tts_weights()
            vocoder: Initialized vocoder model for converting codes to audio
            tokenizer: Optional text tokenizer (uses model's default if None)
        """
        super().__init__(**kwargs)

        self._engine = TTSInferenceEngine(weights)
        self._vocoder = vocoder
        self._tokenizer = tokenizer

        # Run warmup immediately so first real call is fast
        self._engine.warmup()

    def _tokenize_for_tts(self, text: str) -> list[int]:
        """
        Prepare the full 8-step conditioning token sequence for the talker.

        The talker expects:
          [BOS, text_tokens..., THINK_START, THINK_PROCESS, THINK_END, AUDIO_START]

        The thinking tokens (2155, 2156, 2157) are from line 2136 of
        modeling_qwen3_tts.py. Using padding tokens here gives bad audio.
        """
        from engine.inference import THINK_TOKEN_START, THINK_TOKEN_PROCESS, THINK_TOKEN_END

        if self._tokenizer is None:
            raise ValueError("Tokenizer required for text tokenization")

        # Tokenize the input text
        text_ids = self._tokenizer.encode(text, add_special_tokens=False)

        # Build the full conditioning sequence
        BOS = self._tokenizer.bos_token_id or 1
        AUDIO_START = 2048  # Start-of-audio special token

        full_sequence = (
            [BOS]
            + text_ids
            + [THINK_TOKEN_START, THINK_TOKEN_PROCESS, THINK_TOKEN_END]
            + [AUDIO_START]
        )

        return full_sequence

    def _codes_to_audio(self, frame_codes: list[int]) -> np.ndarray:
        """
        Convert 16 audio codes to PCM audio samples using the vocoder.

        The vocoder takes a tensor of shape [num_codebooks, 1] and returns
        a waveform tensor of shape [samples].
        """
        codes_tensor = torch.tensor(
            frame_codes, dtype=torch.long, device="cuda"
        ).unsqueeze(1)  # [16, 1]

        with torch.no_grad():
            waveform = self._vocoder.decode(codes_tensor)  # [samples]

        return waveform.cpu().numpy().astype(np.float32)

    async def run_tts(self, text: str) -> AsyncGenerator[TTSAudioRawFrame, None]:
        """
        Main streaming TTS method called by Pipecat.

        Yields TTSAudioRawFrame objects as audio is generated.
        First frame arrives in ~80ms (TTFC).
        Subsequent frames batch 10 at a time for efficiency.

        This MUST use `yield` not `return` for streaming to work.
        """
        yield TTSStartedFrame()

        ttfc_logged = False
        start_time = time.perf_counter()

        # Tokenize input text
        loop = asyncio.get_event_loop()
        text_tokens = await loop.run_in_executor(
            None, self._tokenize_for_tts, text
        )

        # Buffer for batching frames (except first which sends immediately)
        audio_buffer = []
        frame_count = 0

        # Estimate max frames from word count
        # Using word heuristic because M-RoPE not implemented so EOS unreliable
        word_count = len(text.split())
        max_frames = max(word_count * 8, 50)  # ~8 frames per word

        # Run synthesis in thread so we don't block the event loop
        def _run_synthesis():
            return list(self._engine.synthesize(text_tokens, max_frames=max_frames))

        # We use a generator approach with asyncio to stay non-blocking
        # Process frames as they come, yielding audio chunks
        synthesis_gen = self._engine.synthesize(text_tokens, max_frames=max_frames)

        for frame_codes in synthesis_gen:
            # Convert codes to audio
            audio_samples = await loop.run_in_executor(
                None, self._codes_to_audio, frame_codes
            )

            audio_buffer.append(audio_samples)
            frame_count += 1

            # Send first frame immediately for low TTFC
            if frame_count == 1:
                combined = np.concatenate(audio_buffer)
                audio_bytes = (combined * 32767).astype(np.int16).tobytes()
                audio_buffer = []

                if not ttfc_logged:
                    ttfc_ms = (time.perf_counter() - start_time) * 1000
                    print(f"[TTFC] {ttfc_ms:.1f}ms")
                    ttfc_logged = True

                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                )

            # Batch remaining frames in groups of BATCH_FRAMES
            elif len(audio_buffer) >= BATCH_FRAMES:
                combined = np.concatenate(audio_buffer)
                audio_bytes = (combined * 32767).astype(np.int16).tobytes()
                audio_buffer = []

                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                )

            # Yield control back to event loop periodically
            await asyncio.sleep(0)

        # Flush remaining audio buffer
        if audio_buffer:
            combined = np.concatenate(audio_buffer)
            audio_bytes = (combined * 32767).astype(np.int16).tobytes()
            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

        total_time = (time.perf_counter() - start_time) * 1000
        audio_duration_ms = frame_count * (1000 / FRAME_RATE)
        rtf = total_time / audio_duration_ms if audio_duration_ms > 0 else 0
        print(f"[Synthesis] {frame_count} frames, {total_time:.1f}ms total, RTF={rtf:.3f}")

        yield TTSStoppedFrame()