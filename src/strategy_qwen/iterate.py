"""
Run all Qwen prompt variants sequentially and produce a comparison table.

Usage:
    python -m src.strategy_qwen.iterate [--limit 100] [--variants q1,q2,q3]
"""
import argparse
import json
import sys
from pathlib import Path

from src.strategy_qwen.runner import load_data, run_variant, RESULTS_DIR
from src.strategy_qwen.prompts import VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of trial samples")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variant names (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3.5 thinking mode (slow but may improve reasoning)")
    args = parser.parse_args()

    train, trial = load_data()

    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]
    else:
        variant_names = list(VARIANTS.keys())

    # Thinking mode is slow (~1 sample/min). Use --thinking flag to enable.
    results = []
    for name in variant_names:
        if name not in VARIANTS:
            print(f"Unknown variant: {name}")
            continue
        thinking = args.thinking
        print(f"\n>>> Running variant: {name} ({args.limit} samples, thinking={'ON' if thinking else 'OFF'}) <<<\n")
        metrics = run_variant(name, VARIANTS[name], trial, train, limit=args.limit, seed=args.seed, enable_thinking=thinking)
        results.append(metrics)

    # Comparison table
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Variant':<25s} | {'QWK':>6s} | {'Acc':>6s} | {'Err':>4s} | {'Time':>6s} | {'Correct R':>9s} | {'PartCor R':>9s} | {'Incor R':>7s}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*4}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
    for m in results:
        pc = m.get("per_class", {})
        print(f"  {m['variant']:<25s} | {m.get('qwk', 0):>6.3f} | {m.get('accuracy', 0):>6.1%} | {m['errors']:>4d} | {m['elapsed_s']:>5.0f}s | "
              f"{pc.get('Correct', {}).get('R', 0):>9.3f} | {pc.get('Partially correct', {}).get('R', 0):>9.3f} | {pc.get('Incorrect', {}).get('R', 0):>7.3f}")
    print("=" * 80)

    # Save comparison
    comparison_file = RESULTS_DIR / "comparison_latest.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {comparison_file}")


if __name__ == "__main__":
    main()
