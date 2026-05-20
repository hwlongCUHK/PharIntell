#!/usr/bin/env python3
"""DCV: accuracy + macro-F1 vs datasets/phar_dcv/ground_truth.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lib_phar_eval import PHAR_ROOT

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", "ERROR"]


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    """Per-class F1."""
    per: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for c in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1}
        if any(t == c for t in y_true):
            f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return macro, per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores_path",
        type=str,
        default="scores/phar_dcv",
    )
    ap.add_argument(
        "--ground_truth",
        type=str,
        default="datasets/phar_dcv/ground_truth.json",
    )
    args = ap.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PHAR_ROOT / gt_path
    scores_dir = Path(args.scores_path)
    if not scores_dir.is_absolute():
        scores_dir = PHAR_ROOT / scores_dir

    if not gt_path.is_file():
        raise SystemExit(f"Missing ground truth: {gt_path}")

    gold = json.loads(gt_path.read_text(encoding="utf-8"))

    summary: dict[str, Any] = {}
    for score_file in sorted(scores_dir.glob("*.json")):
        pred = json.loads(score_file.read_text(encoding="utf-8"))
        y_true: list[str] = []
        y_pred: list[str] = []
        for qid, g in gold.items():
            if qid not in pred:
                continue
            y_true.append(str(g.get("label", "ERROR")).upper())
            y_pred.append(str(pred[qid]).upper())

        if not y_true:
            continue
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        mf1, per = macro_f1(y_true, y_pred)
        summary[score_file.name] = {
            "n": len(y_true),
            "accuracy": acc,
            "macro_f1": mf1,
            "per_class": per,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
