"""Evaluate text and vision models through LiteClient.
Reports success rate, execution time, and token usage.

Usage:
    python eval_models.py
    python eval_models.py --mode text
    python eval_models.py --mode vision
    python eval_models.py --models gemma4:e2b gemma4:e4b
    python eval_models.py --runs 3          # repeat each case N times
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import litellm

# ══════════════════════════════════════════════════════════════════
#  Token tracker — wraps litellm.completion BEFORE lite imports it
# ══════════════════════════════════════════════════════════════════

_last_usage: dict = {}

def _tracked_completion(*args, **kwargs):
    resp = _original_completion(*args, **kwargs)
    if hasattr(resp, "usage") and resp.usage:
        _last_usage["prompt"] = getattr(resp.usage, "prompt_tokens", 0)
        _last_usage["completion"] = getattr(resp.usage, "completion_tokens", 0)
        _last_usage["total"] = getattr(resp.usage, "total_tokens", 0)
    return resp

_original_completion = litellm.completion
litellm.completion = _tracked_completion

from pydantic import BaseModel, Field

from litekit.lite_client import LiteClient
from litekit.config import ModelConfig, ModelInput

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


# ============================================================
#  Pydantic schemas for structured-output tests
# ============================================================

class ResearchBrief(BaseModel):
    topic: str = Field(description="The research topic")
    summary: str = Field(description="One-paragraph summary")
    key_findings: list[str] = Field(description="3–5 key findings")
    year_range: str = Field(description="Relevant year range, e.g. '2020–2025'")


class MultiChoiceAnswer(BaseModel):
    question: str = Field(description="The original question")
    selected: str = Field(description="Selected option label (A/B/C/D)")
    explanation: str = Field(description="Why this option is correct")


# ============================================================
#  Test-case definitions
# ============================================================

@dataclass
class EvalCase:
    name: str
    prompt: str
    schema: Optional[type[BaseModel]] = None
    image_path: Optional[str] = None


TEXT_CASES = [
    EvalCase("simple_capital", "What is the capital of France? Answer in one word."),
    EvalCase("simple_entanglement", "Explain quantum entanglement in one sentence."),
    EvalCase("simple_haiku", "Write a haiku about Python programming."),
    EvalCase("pydantic_research",
             "Research the impact of AI on healthcare between 2020 and 2025.",
             ResearchBrief),
    EvalCase("pydantic_mcq",
             "Multiple choice: What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6",
             MultiChoiceAnswer),
]


def _find_images(max_images: int = 2) -> list[str]:
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(Path.cwd().glob(pattern))
        images.extend(Path.cwd().glob(f"**/{pattern}"))
    seen = set()
    chosen = []
    for img in images:
        stem = img.stem.lower()
        if stem in seen or "logo" in stem or "icon" in stem:
            continue
        if img.stat().st_size < 2000:
            continue
        seen.add(stem)
        chosen.append(str(img))
        if len(chosen) >= max_images:
            break
    return chosen


# ============================================================
#  Model lists
# ============================================================

TEXT_MODELS = [
    ("gemma3:latest",   "ollama/gemma3:latest"),
    ("gemma3:12b",      "ollama/gemma3:12b"),
    ("gemma4:e2b",      "ollama/gemma4:e2b"),
    ("gemma4:e4b",      "ollama/gemma4:e4b"),
    ("gemma4:12b",      "ollama/gemma4:12b"),
    ("gemma4:e2b-mlx",  "ollama/gemma4:e2b-mlx"),
    ("gemma4:e4b-mlx",  "ollama/gemma4:e4b-mlx"),
    ("gemma4:12b-mlx",  "ollama/gemma4:12b-mlx"),
]

VISION_MODELS = [
    ("gemma3:12b",      "ollama/gemma3:12b"),
    ("gemma4:e2b",      "ollama/gemma4:e2b"),
    ("gemma4:e4b",      "ollama/gemma4:e4b"),
    ("gemma4:12b",      "ollama/gemma4:12b"),
    ("gemma4:12b-mlx",  "ollama/gemma4:12b-mlx"),
]

# ============================================================
#  Evaluation runner
# ============================================================

@dataclass
class EvalResult:
    model: str
    case: str
    success: bool
    elapsed: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str = ""


def run_cases(model_name: str, model_id: str,
              cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    config = ModelConfig(model=model_id, temperature=0.1, timeout=180)
    client = LiteClient(config)

    for case in cases:
        _last_usage.clear()
        start = time.perf_counter()
        try:
            inp = ModelInput(
                user_prompt=case.prompt,
                response_format=case.schema,
                image_path=case.image_path,
            )
            resp = client.generate_text(inp)
            elapsed = time.perf_counter() - start

            if case.schema:
                success = isinstance(resp, case.schema)
            else:
                success = isinstance(resp, str) and len(resp) > 0

            results.append(EvalResult(
                model=model_name,
                case=case.name,
                success=success,
                elapsed=round(elapsed, 2),
                prompt_tokens=_last_usage.get("prompt", 0),
                completion_tokens=_last_usage.get("completion", 0),
                total_tokens=_last_usage.get("total", 0),
            ))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append(EvalResult(
                model=model_name,
                case=case.name,
                success=False,
                elapsed=round(elapsed, 2),
                error=str(e)[:200],
            ))
    return results


# ============================================================
#  Report formatting
# ============================================================

def print_report(all_results: list[EvalResult], title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in all_results:
        by_model[r.model].append(r)

    grand_total = len(all_results)
    grand_passed = sum(1 for r in all_results if r.success)
    grand_tokens = sum(r.total_tokens for r in all_results)
    grand_time = sum(r.elapsed for r in all_results)

    for model, results in by_model.items():
        passed = sum(1 for r in results if r.success)
        total = len(results)
        times = [r.elapsed for r in results]
        avg_time = statistics.mean(times)
        toks = [r.completion_tokens for r in results if r.completion_tokens > 0]
        avg_tok = statistics.mean(toks) if toks else 0

        print(f"\n  {model}")
        print(f"  {'─'*70}")
        print(f"    {'Case':30s} {'Status':8s} {'Time':>8s}  {'Out Tok':>7s}")
        print(f"    {'─'*30} {'─'*8} {'─'*8}  {'─'*7}")
        for r in results:
            status = "✓ PASS" if r.success else "✗ FAIL"
            tok_str = str(r.completion_tokens) if r.completion_tokens else "—"
            print(f"    {r.case:30s} {status:8s} {r.elapsed:>6.2f}s  {tok_str:>7s}")
            if r.error:
                print(f"    {'':30s} error: {r.error}")
        print(f"    {'─'*70}")
        print(f"    {'TOTAL':30s} {passed:>3d}/{total:<3d}  {avg_time:>6.2f}s  {avg_tok:>5.0f}/call")

    print(f"\n  {'='*70}")
    print(f"  Grand total: {grand_passed}/{grand_total} passed  "
          f"{grand_time:.1f}s elapsed  {grand_tokens} tokens")
    print(f"  {'='*70}\n")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text & vision models through LiteClient",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["text", "vision", "all"],
                        default="all", help="Test category (default: all)")
    parser.add_argument("--models", nargs="*",
                        help="Specific model short-names to test")
    parser.add_argument("--runs", type=int, default=1,
                        help="Repeat each case N times (default: 1)")
    args = parser.parse_args()

    images = _find_images()
    if images:
        VISION_CASES = [
            EvalCase(f"describe_{Path(p).stem}",
                     "Describe this image in detail. What do you see?",
                     image_path=p)
            for p in images
        ]
    else:
        VISION_CASES = []
        if args.mode in ("vision", "all"):
            print("⚠ No suitable images found for vision testing. "
                  "Place a .jpg/.png in the current directory.")

    text_models = TEXT_MODELS
    vision_models = VISION_MODELS
    if args.models:
        text_models = [(n, i) for n, i in text_models if n in args.models]
        vision_models = [(n, i) for n, i in vision_models if n in args.models]
        if not text_models and not vision_models:
            print(f"No matching models for --models {args.models}")
            sys.exit(1)

    all_results: list[EvalResult] = []

    if args.mode in ("text", "all") and text_models:
        print(f"\n● Text: {len(text_models)} models × {len(TEXT_CASES)} cases"
              f"{' × ' + str(args.runs) + ' runs' if args.runs > 1 else ''}")
        for name, mid in text_models:
            for _ in range(args.runs):
                all_results.extend(run_cases(name, mid, TEXT_CASES))

    if args.mode in ("vision", "all") and VISION_CASES and vision_models:
        print(f"\n● Vision: {len(vision_models)} models × {len(VISION_CASES)} images"
              f"{' × ' + str(args.runs) + ' runs' if args.runs > 1 else ''}")
        for name, mid in vision_models:
            for _ in range(args.runs):
                all_results.extend(run_cases(name, mid, VISION_CASES))

    if not all_results:
        print("No tests were run.")
        sys.exit(0)

    text_results = [
        r for r in all_results
        if not any(r.case.startswith(f"describe_{Path(p).stem}")
                   for p in images)
    ] if images else all_results

    vision_results = [
        r for r in all_results
        if any(r.case.startswith(f"describe_{Path(p).stem}")
               for p in images)
    ] if images else []

    if text_results:
        print_report(text_results, "TEXT EVALUATION")
    if vision_results:
        print_report(vision_results, "VISION EVALUATION")


if __name__ == "__main__":
    main()
