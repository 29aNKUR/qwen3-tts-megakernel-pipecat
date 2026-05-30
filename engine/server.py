"""
Streaming inference server for Qwen3-TTS megakernel.

Step 2 of the assessment: expose a simple streaming interface.
  - prompt in
  - token stream out (Server-Sent Events)
  - audio stream out (raw PCM chunks)

This sits between the megakernel engine and Pipecat.
Pipecat calls /synthesize and receives audio chunks as they are produced.
No buffering. First chunk arrives in under 90ms.

Usage:
  export LDG_LM_NUM_BLOCKS=16
  python server.py

  # Test it:
  curl -X POST http://localhost:8000/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, how are you today?"}' \
    --no-buffer
"""

import os
import time
import asyncio
import json
import logging
from typing import AsyncGenerator

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Build megakernel with TTS vocab constants before any imports that touch torch ops
os.environ.setdefault("LDG_LM_NUM_BLOCKS", "16")
from qwen_megakernel.build import get_extension
log.info("Building megakernel extension...")
get_extension()
log.info("Megakernel ready.")

from engine.weights import load_tts_weights
from engine.inference import TTSInferenceEngine

app = FastAPI(title="Qwen3-TTS Megakernel Inference Server")

# Global state - loaded once at startup
_engine: TTSInferenceEngine = None
_vocoder = None
_tokenizer = None


class SynthesizeRequest(BaseModel):
    text: str
    max_frames: int = 512
    stream_tokens: bool = False  # if True, stream raw token IDs instead of audio


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    device: str


@app.on_event("startup")
async def startup():
    """Load all models at server startup. Takes ~60 seconds on first run."""
    global _engine, _vocoder, _tokenizer

    log.info("Loading Qwen3-TTS weights...")
    weights = load_tts_weights("Qwen/Qwen3-TTS", verbose=True)

    log.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(weights["model_path"])

    log.info("Loading vocoder...")
    _vocoder = _load_vocoder(weights["model_path"])

    log.info("Initializing inference engine...")
    _engine = TTSInferenceEngine(weights, verbose=True)

    log.info("Running warmup...")
    _engine.warmup(verbose=True)

    log.info("Server ready. Listening for requests.")


def _load_vocoder(model_path: str):
    """Load the Qwen3-TTS vocoder for converting audio codes to waveforms."""
    import sys
    sys.path.insert(0, model_path)
    try:
        from modeling_qwen3_tts import Qwen3TTSVocoder
        vocoder = Qwen3TTSVocoder.from_pretrained(
            os.path.join(model_path, "vocoder")
        ).cuda().eval()
        log.info("Vocoder loaded successfully.")
        return vocoder
    except Exception as e:
        log.warning(f"Could not load vocoder: {e}. Audio output will be unavailable.")
        return None


def _tokenize(text: str) -> list[int]:
    """
    Build the full conditioning token sequence for the talker decoder.

    Sequence: [BOS, text_tokens..., THINK_START, THINK_PROCESS, THINK_END, AUDIO_START]

    The three thinking tokens (2155, 2156, 2157) are from line 2136 of
    modeling_qwen3_tts.py. Critical for audio quality.
    """
    from engine.inference import THINK_TOKEN_START, THINK_TOKEN_PROCESS, THINK_TOKEN_END

    text_ids = _tokenizer.encode(text, add_special_tokens=False)
    BOS = _tokenizer.bos_token_id or 1
    AUDIO_START = 2048

    return (
        [BOS]
        + text_ids
        + [THINK_TOKEN_START, THINK_TOKEN_PROCESS, THINK_TOKEN_END]
        + [AUDIO_START]
    )


def _codes_to_audio(frame_codes: list[int]) -> bytes:
    """Convert 16 audio codes to raw PCM bytes using the vocoder."""
    if _vocoder is None:
        # Return silence if vocoder not available
        return bytes(3840)  # 1920 samples * 2 bytes (int16)

    codes_tensor = torch.tensor(
        frame_codes, dtype=torch.long, device="cuda"
    ).unsqueeze(1)  # [16, 1]

    with torch.no_grad():
        waveform = _vocoder.decode(codes_tensor)  # [samples]

    samples = waveform.cpu().numpy().astype(np.float32)
    return (samples * 32767).astype(np.int16).tobytes()


async def _stream_audio(text: str, max_frames: int) -> AsyncGenerator[bytes, None]:
    """
    Core streaming generator.

    Yields audio chunks as SSE (Server-Sent Events) as each frame is produced.
    First chunk arrives in under 90ms after warmup.

    Format: each SSE event contains a JSON object:
      - type: "audio" | "metric" | "done"
      - data: base64 audio bytes (for audio events)
      - ttfc_ms: time to first chunk in ms (for first audio event)
      - rtf: real-time factor (for done event)
      - total_frames: total frames generated (for done event)
    """
    import base64

    text_tokens = _tokenize(text)
    loop = asyncio.get_event_loop()

    start_time = time.perf_counter()
    first_chunk = True
    frame_count = 0

    for frame_codes in _engine.synthesize(text_tokens, max_frames=max_frames):
        # Convert codes to audio in thread pool (non-blocking)
        audio_bytes = await loop.run_in_executor(
            None, _codes_to_audio, frame_codes
        )

        frame_count += 1
        now = time.perf_counter()

        if first_chunk:
            ttfc_ms = (now - start_time) * 1000
            log.info(f"TTFC: {ttfc_ms:.1f}ms")

            event = {
                "type": "audio",
                "data": base64.b64encode(audio_bytes).decode(),
                "ttfc_ms": round(ttfc_ms, 2),
                "frame": frame_count,
            }
            first_chunk = False
        else:
            event = {
                "type": "audio",
                "data": base64.b64encode(audio_bytes).decode(),
                "frame": frame_count,
            }

        yield f"data: {json.dumps(event)}\n\n"

        # Yield control back to event loop
        await asyncio.sleep(0)

    # Send final metrics event
    total_ms = (time.perf_counter() - start_time) * 1000
    audio_duration_ms = frame_count * 80  # 80ms per frame at 12.5fps
    rtf = total_ms / audio_duration_ms if audio_duration_ms > 0 else 0

    log.info(f"Done: {frame_count} frames, {total_ms:.1f}ms, RTF={rtf:.3f}")

    done_event = {
        "type": "done",
        "total_frames": frame_count,
        "total_ms": round(total_ms, 2),
        "audio_duration_ms": round(audio_duration_ms, 2),
        "rtf": round(rtf, 4),
    }
    yield f"data: {json.dumps(done_event)}\n\n"


async def _stream_tokens(text: str, max_frames: int) -> AsyncGenerator[bytes, None]:
    """
    Stream raw token IDs instead of audio. Useful for debugging.

    Each SSE event contains frame_codes: list of 16 audio code integers.
    """
    text_tokens = _tokenize(text)
    start_time = time.perf_counter()
    frame_count = 0

    for frame_codes in _engine.synthesize(text_tokens, max_frames=max_frames):
        frame_count += 1
        now = time.perf_counter()

        event = {
            "type": "tokens",
            "frame": frame_count,
            "codes": frame_codes,
            "elapsed_ms": round((now - start_time) * 1000, 2),
        }
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'type': 'done', 'total_frames': frame_count})}\n\n"


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _engine is not None else "loading",
        engine_ready=_engine is not None,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Main synthesis endpoint.

    Returns a streaming response of Server-Sent Events.
    Each event is a JSON object with type "audio", "metric", or "done".

    Audio events contain base64-encoded raw PCM audio (24kHz, mono, int16).
    The done event contains final performance metrics.

    Example:
      curl -X POST http://localhost:8000/synthesize \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world"}' \
        --no-buffer
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready yet. Try again in a few seconds.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    log.info(f"Synthesizing: '{request.text[:50]}...' " if len(request.text) > 50 else f"Synthesizing: '{request.text}'")

    generator = (
        _stream_tokens(request.text, request.max_frames)
        if request.stream_tokens
        else _stream_audio(request.text, request.max_frames)
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.post("/synthesize/raw")
async def synthesize_raw(request: SynthesizeRequest):
    """
    Raw audio endpoint - streams raw PCM bytes directly.

    No SSE wrapping. Just raw int16 PCM audio bytes streamed continuously.
    Useful for piping directly to audio output.

    Example:
      curl -X POST http://localhost:8000/synthesize/raw \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world"}' \
        --no-buffer | aplay -r 24000 -f S16_LE -c 1
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready yet.")

    text_tokens = _tokenize(request.text)
    loop = asyncio.get_event_loop()

    async def raw_audio_generator():
        for frame_codes in _engine.synthesize(text_tokens, max_frames=request.max_frames):
            audio_bytes = await loop.run_in_executor(
                None, _codes_to_audio, frame_codes
            )
            yield audio_bytes
            await asyncio.sleep(0)

    return StreamingResponse(
        raw_audio_generator(),
        media_type="audio/pcm",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )