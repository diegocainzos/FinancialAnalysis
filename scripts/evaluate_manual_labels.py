#!/usr/bin/env python3
"""Evaluate manual labels against FinBERT predictions.

Computes:
- Accuracy
- Recall (macro + weighted + por clase)
- F1 score (macro + weighted + por clase)
- Confusion matrix (manual -> predicción)
Optionally writes a JSON report for history tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("negative", "neutral", "positive")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate manual labels vs FinBERT")
    parser.add_argument(
        "--csv",
        default="data/manual_labeling_finbert.csv",
        help="CSV exported for manual labeling",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write a JSON evaluation report",
    )
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("CSV is empty")

    labeled_rows = []
    invalid_rows = []
    for index, row in enumerate(rows, start=2):
        manual = (row.get("manual_label") or "").strip().lower()
        pred = (row.get("finbert_label") or "").strip().lower()

        if not manual:
            continue
        if manual not in LABELS or pred not in LABELS:
            invalid_rows.append((index, manual, pred))
            continue
        labeled_rows.append((manual, pred))

    if invalid_rows:
        print("Invalid rows found (line, manual_label, finbert_label):")
        for item in invalid_rows[:20]:
            print(item)
        raise SystemExit("Fix invalid labels before evaluating")

    if not labeled_rows:
        raise SystemExit("No manual labels found in 'manual_label' column")

    total = len(labeled_rows)
    correct = sum(1 for manual, pred in labeled_rows if manual == pred)
    accuracy = _safe_div(correct, total)

    # confusion[manual][pred]
    confusion: dict[str, dict[str, int]] = {
        manual: {pred: 0 for pred in LABELS} for manual in LABELS
    }
    for manual, pred in labeled_rows:
        confusion[manual][pred] += 1

    per_class_metrics: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        fp = sum(confusion[m][label] for m in LABELS if m != label)

        recall = _safe_div(tp, tp + fn)
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = sum(confusion[label].values())

        per_class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_recall = sum(float(per_class_metrics[l]["recall"]) for l in LABELS) / len(LABELS)
    macro_f1 = sum(float(per_class_metrics[l]["f1"]) for l in LABELS) / len(LABELS)
    weighted_recall = _safe_div(
        sum(float(per_class_metrics[l]["recall"]) * int(per_class_metrics[l]["support"]) for l in LABELS),
        total,
    )
    weighted_f1 = _safe_div(
        sum(float(per_class_metrics[l]["f1"]) * int(per_class_metrics[l]["support"]) for l in LABELS),
        total,
    )

    print(f"Total manually labeled rows: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (macro): {macro_recall:.4f}")
    print(f"Recall (weighted): {weighted_recall:.4f}")
    print(f"F1 score (macro): {macro_f1:.4f}")
    print(f"F1 score (weighted): {weighted_f1:.4f}")

    print("\nPer-class metrics:")
    for label in LABELS:
        metric = per_class_metrics[label]
        print(
            f"- {label}: precision={float(metric['precision']):.4f} "
            f"recall={float(metric['recall']):.4f} f1={float(metric['f1']):.4f} "
            f"support={int(metric['support'])}"
        )

    print("\nConfusion matrix (manual -> finbert):")
    print("manual\\pred," + ",".join(LABELS))
    for manual in LABELS:
        counts = [str(confusion[manual][pred]) for pred in LABELS]
        print(f"{manual}," + ",".join(counts))

    if args.output_json:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_csv": str(path),
            "total_labeled_rows": total,
            "metrics": {
                "accuracy": accuracy,
                "recall_macro": macro_recall,
                "recall_weighted": weighted_recall,
                "f1_macro": macro_f1,
                "f1_weighted": weighted_f1,
            },
            "per_class": per_class_metrics,
            "confusion_matrix": confusion,
            "labels": list(LABELS),
        }
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {output}")


if __name__ == "__main__":
    main()
