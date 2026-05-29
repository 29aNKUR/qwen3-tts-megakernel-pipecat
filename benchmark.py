"""
Benchmark script for the Qwen3-TTS megakernel engine.

Measures:
  - Tokens per second from the megakernel (talker + code predictor)
  - TTFC (time to first audio chunk) in milliseconds
  - RTF (real-time factor)
  - End-to-end latency breakdown

All timings use torch.cuda.synchronize() barriers to ensure
accurate GPU-side measurements. No hand-waving.

Usage:
  python benchmark.py
  python benchmark.py --text "Hello, how are you today?"
  python benchmark.py --runs 5
"""

import os
import time
import argparse
import statistics

import torch

# Build megakernel with TTS constants
os.environ["LDG_LM_NUM_BLOCKS"] = "16"
from qwen_megakernel.build import get_extension
get_extension()

from engine.weights import load_tts_weights, TALKER_NUM_LAYERS, PREDICTOR_NUM_LAYERS
from engine.inference import TTSInferenceEngine

FRAME_RATE = 12.5  # frames per second
MS_PER_FRAME = 1000 / FRAME_RATE  # 80ms of audio per frame
SAMPLE_RATE = 24000


def benchmark_megakernel_raw(engine: TTSInferenceEngine, num_steps: int = 100):
    """
    Benchmark raw megakernel throughput (tokens per second).
    Tests both talker (28 layers) and code predictor (5 layers).
    """
    print(f"\n{'='*50}")
    print("RAW MEGAKERNEL THROUGHPUT")
    print(f"{'='*50}")

    # Warm up (should already be done but just in case)
    for _ in range(5):
        engine._talker_step(0)
    engine._talker_position = 0
    engine._talker_k_cache.zero_()
    engine._talker_v_cache.zero_()
    torch.cuda.synchronize()

    # Benchmark talker (28 layers)
    start = time.perf_counter()
    torch.cuda.synchronize()

    for i in range(num_steps):
        engine._talker_step(i % 100)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    talker_tok_s = num_steps / elapsed
    talker_ms_tok = elapsed * 1000 / num_steps
    print(f"Talker (28 layers):        {talker_tok_s:.1f} tok/s  ({talker_ms_tok:.2f} ms/tok)")

    # Reset
    engine._talker_position = 0
    engine._talker_k_cache.zero_()
    engine._talker_v_cache.zero_()

    # Benchmark code predictor (5 layers)
    for _ in range(5):
        engine._predictor_step(0)
    engine._predictor_position = 0
    engine._predictor_k_cache.zero_()
    engine._predictor_v_cache.zero_()
    torch.cuda.synchronize()

    start = time.perf_counter()
    torch.cuda.synchronize()

    for i in range(num_steps):
        engine._predictor_step(i % 100)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    predictor_tok_s = num_steps / elapsed
    predictor_ms_tok = elapsed * 1000 / num_steps
    print(f"Code predictor (5 layers): {predictor_tok_s:.1f} tok/s  ({predictor_ms_tok:.2f} ms/tok)")

    # Reset
    engine._predictor_position = 0
    engine._predictor_k_cache.zero_()
    engine._predictor_v_cache.zero_()

    return talker_tok_s, predictor_tok_s


def benchmark_ttfc_rtf(engine: TTSInferenceEngine, text_tokens: list, num_runs: int = 3):
    """
    Benchmark TTFC and RTF across multiple runs.

    TTFC = time from synthesis start to first frame yielded
    RTF  = total synthesis time / total audio duration
    """
    print(f"\n{'='*50}")
    print("TTFC AND RTF BENCHMARK")
    print(f"{'='*50}")

    ttfc_results = []
    rtf_results = []
    frame_counts = []

    for run in range(num_runs):
        torch.cuda.synchronize()
        run_start = time.perf_counter()
        first_frame_time = None
        frame_count = 0

        for frame_codes in engine.synthesize(text_tokens, max_frames=100):
            torch.cuda.synchronize()
            now = time.perf_counter()

            if first_frame_time is None:
                first_frame_time = now
                ttfc_ms = (first_frame_time - run_start) * 1000
                ttfc_results.append(ttfc_ms)

            frame_count += 1

        total_time_ms = (time.perf_counter() - run_start) * 1000
        audio_duration_ms = frame_count * MS_PER_FRAME
        rtf = total_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0

        rtf_results.append(rtf)
        frame_counts.append(frame_count)

        print(f"  Run {run+1}: TTFC={ttfc_results[-1]:.1f}ms  RTF={rtf:.3f}  frames={frame_count}")

    print(f"\n  TTFC avg: {statistics.mean(ttfc_results):.1f}ms  (target <90ms)")
    print(f"  TTFC min: {min(ttfc_results):.1f}ms")
    print(f"  RTF  avg: {statistics.mean(rtf_results):.3f}  (target <0.3)")
    print(f"  RTF  min: {min(rtf_results):.3f}")

    passed_ttfc = statistics.mean(ttfc_results) < 90
    passed_rtf = statistics.mean(rtf_results) < 0.3

    print(f"\n  TTFC target (<90ms):  {'PASS' if passed_ttfc else 'FAIL'}")
    print(f"  RTF  target (<0.3):   {'PASS' if passed_rtf else 'FAIL'}")

    return ttfc_results, rtf_results


def benchmark_per_frame_breakdown(engine: TTSInferenceEngine, text_tokens: list):
    """
    Detailed per-component timing breakdown for one synthesis run.
    Shows exactly where time is spent.
    """
    print(f"\n{'='*50}")
    print("PER-FRAME TIMING BREAKDOWN")
    print(f"{'='*50}")

    engine._reset()
    engine._prefill(text_tokens)

    talker_times = []
    predictor_times = []
    num_frames = 20  # measure first 20 frames

    current_codes = [0] * 16

    for frame_idx in range(num_frames):
        # Time talker step
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if frame_idx == 0:
            talker_code = engine._talker_step(1)
        else:
            engine._compute_combined_embedding(current_codes)
            talker_code = engine._talker_step(-1)

        torch.cuda.synchronize()
        talker_times.append((time.perf_counter() - t0) * 1000)

        # Time code predictor (15 steps)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        frame_codes = [talker_code]
        predictor_input = talker_code
        for _ in range(15):
            predictor_code = engine._predictor_step(predictor_input)
            frame_codes.append(predictor_code)
            predictor_input = predictor_code

        torch.cuda.synchronize()
        predictor_times.append((time.perf_counter() - t0) * 1000)

        current_codes = frame_codes

    avg_talker = statistics.mean(talker_times)
    avg_predictor = statistics.mean(predictor_times)
    avg_total = avg_talker + avg_predictor

    print(f"  Talker step (1 token):        {avg_talker:.2f}ms avg")
    print(f"  Code predictor (15 tokens):   {avg_predictor:.2f}ms avg")
    print(f"  Total per frame:              {avg_total:.2f}ms avg")
    print(f"  Frames per second possible:   {1000/avg_total:.1f}")
    print(f"  (need {FRAME_RATE} fps for real-time)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS megakernel")
    parser.add_argument(
        "--text",
        default="Hello, how are you today? I hope everything is going well.",
        help="Text to synthesize for TTFC/RTF benchmarks"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    args = parser.parse_args()

    print("Loading weights...")
    weights = load_tts_weights("Qwen/Qwen3-TTS", verbose=False)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights["model_path"])

    print("Initializing engine...")
    engine = TTSInferenceEngine(weights, verbose=False)

    print("Running warmup...")
    engine.warmup(verbose=True)

    # Tokenize test text
    text_tokens = tokenizer.encode(args.text, add_special_tokens=True)
    print(f"\nTest text: '{args.text}'")
    print(f"Token count: {len(text_tokens)}")

    # Run benchmarks
    benchmark_megakernel_raw(engine)
    benchmark_per_frame_breakdown(engine, text_tokens)
    benchmark_ttfc_rtf(engine, text_tokens, num_runs=args.runs)

    print(f"\n{'='*50}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()