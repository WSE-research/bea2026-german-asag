"""
Qwen iteration runner — tests prompt variants on N trial samples.

Usage:
    python -m src.strategy_qwen.runner --variant q1 --limit 100
"""
import argparse
import json
import logging
import time
import importlib
from collections import Counter
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "strategy_qwen"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_3WAY = PROJECT_ROOT / "data" / "raw" / "3way"
TRAIN_FILE = DATA_3WAY / "ALICE_LP_train_3way__v2.json"
TRIAL_FILE = DATA_3WAY / "ALICE_LP_trial_3way__v2.json"

VLLM_URL = "http://localhost:8081/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-27B-FP8"

LABELS = ["Correct", "Partially correct", "Incorrect"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    with open(TRAIN_FILE) as f:
        train = json.load(f)
    with open(TRIAL_FILE) as f:
        trial = json.load(f)
    return train, trial


def call_model(messages, max_tokens=300, temperature=0.2, enable_thinking=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Qwen3.5 has built-in thinking mode that wastes tokens.
    # Disable it for direct JSON scoring; enable it for CoT variants.
    if not enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    resp = httpx.post(VLLM_URL, json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_score(text):
    """Extract score from model response. Handles JSON, markdown, and plain text."""
    text = text.strip()

    # Try JSON parse
    try:
        obj = json.loads(text)
        return obj.get("score", obj.get("label", obj.get("bewertung")))
    except json.JSONDecodeError:
        pass

    # Try markdown-fenced JSON
    if "```" in text:
        try:
            inner = text.split("```")[1]
            if inner.startswith("json"):
                inner = inner[4:]
            obj = json.loads(inner.strip())
            return obj.get("score", obj.get("label"))
        except (json.JSONDecodeError, IndexError):
            pass

    # Try finding JSON object in text
    brace = text.rfind("{")
    if brace >= 0:
        try:
            obj = json.loads(text[brace:])
            return obj.get("score", obj.get("label"))
        except json.JSONDecodeError:
            pass

    # Plain text matching
    text_lower = text.lower()
    if "incorrect" in text_lower and "partially" not in text_lower:
        return "Incorrect"
    if "partially correct" in text_lower or "partially" in text_lower:
        return "Partially correct"
    if "correct" in text_lower:
        return "Correct"

    return None


def compute_metrics(golds, preds):
    g = [LABEL_MAP[x] for x in golds]
    p = [LABEL_MAP[x] for x in preds]
    qwk = cohen_kappa_score(g, p, weights="quadratic")
    acc = sum(1 for a, b in zip(golds, preds) if a == b) / len(golds)

    per_class = {}
    for label in LABELS:
        tp = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl == label)
        fp = sum(1 for gl, pl in zip(golds, preds) if gl != label and pl == label)
        fn = sum(1 for gl, pl in zip(golds, preds) if gl == label and pl != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[label] = {"P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3),
                            "support": sum(1 for x in golds if x == label)}

    confusion = {}
    for g_label in LABELS:
        row = {}
        for p_label in LABELS:
            row[p_label] = sum(1 for gl, pl in zip(golds, preds) if gl == g_label and pl == p_label)
        confusion[g_label] = row

    return {"qwk": round(qwk, 4), "accuracy": round(acc, 4), "per_class": per_class, "confusion": confusion}


def run_variant(variant_name, build_messages_fn, trial, train, limit=100, seed=42, enable_thinking=False):
    """Run a prompt variant on trial samples and return metrics."""
    import random
    random.seed(seed)
    samples = trial[:limit]

    # Thinking mode generates reasoning tokens — need higher max_tokens
    max_tok = 2000 if enable_thinking else 300

    golds, preds = [], []
    errors = 0
    raw_results = []

    start = time.time()
    for i, sample in enumerate(samples):
        try:
            messages = build_messages_fn(sample, train)
            response = call_model(messages, max_tokens=max_tok, enable_thinking=enable_thinking)
            score = parse_score(response)

            if score not in LABELS:
                errors += 1
                raw_results.append({"id": sample["id"], "error": f"Invalid score: {score}", "raw": response[:200]})
                continue

            golds.append(sample["score"])
            preds.append(score)
            raw_results.append({"id": sample["id"], "gold": sample["score"], "pred": score, "match": score == sample["score"]})

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                logger.info(f"  [{i+1:3d}/{limit}] {(i+1)/elapsed:.1f} samples/s | errors: {errors}")
        except Exception as e:
            errors += 1
            raw_results.append({"id": sample["id"], "error": str(e)})

    elapsed = time.time() - start
    metrics = compute_metrics(golds, preds) if golds else {"error": "No valid predictions"}
    metrics["errors"] = errors
    metrics["total"] = len(samples)
    metrics["scored"] = len(golds)
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["samples_per_s"] = round(len(samples) / elapsed, 2)
    metrics["variant"] = variant_name
    metrics["model"] = MODEL

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_file = RESULTS_DIR / f"metrics_{variant_name}_{timestamp}.json"
    preds_file = RESULTS_DIR / f"predictions_{variant_name}_{timestamp}.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(preds_file, "w") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {variant_name} | {MODEL}")
    print(f"{'='*60}")
    print(f"  QWK: {metrics.get('qwk', 'N/A')} | Acc: {metrics.get('accuracy', 'N/A')} | Errors: {errors}/{len(samples)}")
    print(f"  Time: {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")
    if "per_class" in metrics:
        for label, stats in metrics["per_class"].items():
            print(f"  {label:>20s}: P={stats['P']:.3f} R={stats['R']:.3f} F1={stats['F1']:.3f} (n={stats['support']})")
    if "confusion" in metrics:
        print(f"  Confusion (rows=gold, cols=pred):")
        print(f"  {'':>20s} | {'Correct':>8s} | {'Part.cor':>8s} | {'Incorrect':>9s}")
        for g_label in LABELS:
            row = metrics["confusion"][g_label]
            print(f"  {g_label:>20s} | {row['Correct']:>8d} | {row['Partially correct']:>8d} | {row['Incorrect']:>9d}")
    print(f"  Saved: {metrics_file.name}")
    print(f"{'='*60}\n")

    return metrics
