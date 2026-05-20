#!/usr/bin/env python3
"""PhV-ADE: classification metrics + simple AE overlap on ADE-positive gold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lib_phar_eval import PHAR_ROOT


def macro_f1_tri(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    classes = ["ADE-positive", "ADE-negative", "Unclear"]
    per: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1}
        if any(t == c for t in y_true):
            f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per


def norm_set(xs: list) -> set[str]:
    out: set[str] = set()
    for x in xs or []:
        if isinstance(x, str) and x.strip():
            out.add(x.strip().lower())
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", type=str, default="scores/phar_phv_ade")
    ap.add_argument(
        "--ground_truth",
        type=str,
        default="datasets/phar_phv_ade/ground_truth.json",
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
        jac_scores: list[float] = []

        for qid, g in gold.items():
            if qid not in pred:
                continue
            t_cls = str(g.get("ade_class", "Unclear"))
            p_row = pred[qid]
            p_cls = str(p_row.get("ade_class", "Unclear"))
            y_true.append(t_cls)
            y_pred.append(p_cls)

            if t_cls == "ADE-positive" and p_cls == "ADE-positive":
                g_ae = norm_set(g.get("adverse_events", []))
                p_ae = norm_set(p_row.get("adverse_events", []))
                jac_scores.append(jaccard(g_ae, p_ae))

        if not y_true:
            continue
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        mf1, per = macro_f1_tri(y_true, y_pred)
        summary[score_file.name] = {
            "n": len(y_true),
            "accuracy": acc,
            "macro_f1": mf1,
            "per_class": per,
            "mean_jaccard_adverse_events_ade_positive_gold": (
                sum(jac_scores) / len(jac_scores) if jac_scores else None
            ),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
