"""
Strategy B: Rubric + Strict Rules Scoring
Usage:
    python -m src.strategy_b_rubric_rules.run [--split train|trial] [--workers 5] [--limit N]
"""

import argparse
import json
import logging
from pathlib import Path

from src.common.data_loader import load_train_3way, load_trial_3way
from src.common.batch_runner import run_batch, compile_submission
from src.common.evaluate import compute_metrics, print_evaluation_report
from src.strategy_b_rubric_rules.scorer import score_sample


def main():
    parser = argparse.ArgumentParser(description="Strategy B: Rubric + Rules Scoring")
    parser.add_argument("--split", choices=["train", "trial"], default="trial")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Limit to N samples (for testing)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data
    data = load_trial_3way() if args.split == "trial" else load_train_3way()
    if args.limit:
        data = data[:args.limit]

    # Output paths
    output_dir = Path("results/strategy_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.split}_results.jsonl"

    # Run batch scoring
    results = run_batch(
        data=data,
        scorer_fn=score_sample,
        output_path=results_path,
        max_workers=args.workers,
        resume=not args.no_resume,
    )

    # Evaluate
    y_true = [r["true_label"] for r in results if r.get("true_label") and not r.get("error")]
    y_pred = [r["score"] for r in results if r.get("true_label") and not r.get("error")]

    if y_true:
        metrics = compute_metrics(y_true, y_pred)
        print_evaluation_report(metrics, title="Strategy B: Rubric + Rules")

        # Save metrics
        with open(output_dir / f"{args.split}_metrics.json", "w") as f:
            json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    # Compile submission
    compile_submission(results_path, output_dir / f"{args.split}_submission_3way.json", track="3way")
    compile_submission(results_path, output_dir / f"{args.split}_submission_2way.json", track="2way")


if __name__ == "__main__":
    main()
