"""
Full voice agent pipeline: STT -> LLM -> Megakernel TTS -> Audio output.

Run this after renting the RTX 5090 on Vast.ai.

Requires:
  - DEEPGRAM_API_KEY env var (free tier available at deepgram.com)
  - OPENAI_API_KEY env var (or swap for any other LLM)
  - RTX 5090 with CUDA 12.8+
  - Qwen3-TTS weights downloaded

Usage:
  python pipeline.py
  python pipeline.py --text-only   # test without microphone
"""

import asyncio
import os
import sys
import argparse

import torch

# Build the megakernel extension first (compiles CUDA with TTS constants)
# Set environment vars BEFORE importing build module
os.environ["LDG_LM_NUM_BLOCKS"] = "16"   # 3072 vocab needs fewer blocks than 151936

from qwen_megakernel.build import get_extension
print("Building megakernel extension with TTS constants...")
get_extension()
print("Megakernel ready.")

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.transports.local.audio import LocalAudioTransportParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from engine.weights import load_tts_weights
from pipecat_service.tts_service import MegakernelTTSService


def load_vocoder(model_path: str):
    """Load the Qwen3-TTS vocoder model."""
    import sys
    sys.path.insert(0, model_path)

    # The vocoder is a lightweight non-DiT model included with Qwen3-TTS
    # It converts 16 audio codes per frame into PCM waveforms
    try:
        from modeling_qwen3_tts import Qwen3TTSVocoder
        vocoder = Qwen3TTSVocoder.from_pretrained(
            os.path.join(model_path, "vocoder")
        ).cuda().eval()
        print("Vocoder loaded.")
        return vocoder
    except ImportError:
        print("Could not load vocoder from model path. Using stub.")
        return None


async def run_pipeline(text_only: bool = False):
    """Build and run the full voice pipeline."""

    # --- Load weights ---
    print("Loading Qwen3-TTS weights...")
    weights = load_tts_weights("Qwen/Qwen3-TTS", verbose=True)
    model_path = weights["model_path"]

    # --- Load vocoder ---
    vocoder = load_vocoder(model_path)

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- Set up Pipecat services ---
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not deepgram_key:
        raise ValueError("Set DEEPGRAM_API_KEY environment variable")
    if not openai_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    stt = DeepgramSTTService(api_key=deepgram_key)

    llm = OpenAILLMService(
        api_key=openai_key,
        model="gpt-4o-mini",  # fast and cheap for demo
    )

    tts = MegakernelTTSService(
        weights=weights,
        vocoder=vocoder,
        tokenizer=tokenizer,
    )

    # System prompt for the voice agent
    context = OpenAILLMContext(
        messages=[{
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Keep responses concise and conversational. "
                "Aim for 1-2 sentences per response."
            )
        }]
    )
    context_aggregator = llm.create_context_aggregator(context)

    if text_only:
        # Text-only mode for testing without microphone
        print("\nText-only mode. Type your message and press Enter.")
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            print("Synthesizing...")
            async for frame in tts.run_tts(user_input):
                pass  # In text-only mode we just benchmark, no audio output
        return

    # --- Full voice pipeline ---
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_audio_passthrough=True,
        )
    )

    pipeline = Pipeline([
        transport.input(),           # Mic input
        stt,                         # Deepgram speech-to-text
        context_aggregator.user(),   # Add user turn to context
        llm,                         # OpenAI LLM
        tts,                         # Megakernel TTS (our custom service)
        transport.output(),          # Speaker output
        context_aggregator.assistant(), # Add assistant turn to context
    ])

    task = PipelineTask(pipeline)

    print("\nVoice agent ready. Speak into your microphone.")
    print("Press Ctrl+C to stop.\n")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Megakernel Voice Agent")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run in text-only mode (no microphone needed)"
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(text_only=args.text_only))