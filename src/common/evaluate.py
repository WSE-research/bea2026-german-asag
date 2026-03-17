"""
Evaluation utilities for the BEA 2026 German ASAG shared task.

Computes Quadratic Weighted Kappa (QWK), accuracy, F1, per-question
breakdowns, and pretty-printed reports. All metrics follow the official
BEA 2026 evaluation protocol where QWK is the primary metric.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_TO_INT = {"Correct": 2, "Partially correct": 1, "Incorrect": 0}


def _labels_to_ints(labels: list[str]) -> list[int]:
    """Convert string labels to ordinal integers.

    Raises:
        ValueError: If an unknown label is encountered.
    """
    ints = []
    for label in labels:
        if label not in LABEL_TO_INT:
            raise ValueError(
                f"Unknown label '{label}'. Expected one of {list(LABEL_TO_INT.keys())}"
            )
        ints.append(LABEL_TO_INT[label])
    return ints


def compute_qwk(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Quadratic Weighted Kappa between true and predicted labels.

    Labels are converted to ordinal integers (Correct=2, Partially correct=1,
    Incorrect=0) before computing QWK, which is appropriate because the labels
    have an inherent ordering.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.

    Returns:
        QWK score as a float in [-1, 1].
    """
    y_true_int = _labels_to_ints(y_true)
    y_pred_int = _labels_to_ints(y_pred)
    return float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic"))


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute a full suite of evaluation metrics.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings.

    Returns:
        Dict with keys:
            - ``qwk``: Quadratic Weighted Kappa (float)
            - ``accuracy``: Overall accuracy (float)
            - ``weighted_f1``: Weighted-average F1 (float)
            - ``classification_report``: Formatted text report (str)
            - ``confusion_matrix``: Confusion matrix as nested list
            - ``per_label``: Per-label precision/recall/F1 dict
    """
    qwk = compute_qwk(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    w_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0))

    report_str = classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    per_label = {}
    for i, label in enumerate(LABELS):
        per_label[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "qwk": qwk,
        "accuracy": acc,
        "weighted_f1": w_f1,
        "classification_report": report_str,
        "confusion_matrix": cm,
        "per_label": per_label,
    }


def per_question_qwk(results: list[dict]) -> pd.DataFrame:
    """Compute QWK and accuracy per question.

    Args:
        results: List of dicts, each with ``question_id``, ``true_label``,
                 and ``pred_label`` keys.

    Returns:
        DataFrame with columns: ``question_id``, ``n``, ``qwk``, ``accuracy``.
        Sorted by QWK ascending (worst questions first) for easy diagnosis.
    """
    df = pd.DataFrame(results)

    rows = []
    for qid, group in df.groupby("question_id"):
        y_true = group["true_label"].tolist()
        y_pred = group["pred_label"].tolist()
        n = len(y_true)

        # QWK requires at least 2 samples and variation in labels
        try:
            qwk = compute_qwk(y_true, y_pred)
        except Exception:
            qwk = float("nan")

        acc = float(accuracy_score(y_true, y_pred))

        rows.append({
            "question_id": qid,
            "n": n,
            "qwk": round(qwk, 4),
            "accuracy": round(acc, 4),
        })

    result_df = pd.DataFrame(rows).sort_values("qwk", ascending=True).reset_index(drop=True)
    logger.info("Computed per-question QWK for %d questions", len(result_df))
    return result_df


def print_evaluation_report(metrics: dict, title: str = "Evaluation Report") -> None:
    """Pretty-print a full evaluation report to stdout.

    Args:
        metrics: Dict as returned by ``compute_metrics``.
        title: Header line for the report.
    """
    width = 60
    sep = "=" * width

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  QWK (primary):    {metrics['qwk']:.4f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"{sep}")
    print()
    print(metrics["classification_report"])
    print()

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    print("Confusion Matrix (rows=true, cols=predicted):")
    header = "                  " + "  ".join(f"{l:>16s}" for l in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row_vals = "  ".join(f"{v:>16d}" for v in cm[i])
        print(f"  {label:>16s}  {row_vals}")
    print()
    print(sep)
