#!/usr/bin/env python3
"""Aggregate per-task eval metrics and gold label distributions for paper tables."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lib_phar_eval import PHAR_ROOT

TASK_GOLD = {
    "phar_dcv": ("datasets/phar_dcv/ground_truth.json", "label"),
    "phar_phv_ade": ("datasets/phar_phv_ade/ground_truth.json", "ade_class"),
    "phar_ccbi": ("datasets/phar_ccbi/ground_truth.json", None),
    "phar_dts_symptom": ("datasets/phar_dts_symptom/ground_truth.json", None),
    "phar_dts_treatment": ("datasets/phar_dts_treatment/ground_truth.json", None),
    "phar_dts_need": ("datasets/phar_dts_need/ground_truth.json", None),
}


def _gold_label_dist(gold_path: Path, label_key: str | None) -> dict[str, int]:
    if not gold_path.is_file():
        return {}
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if label_key:
        counts: Counter[str] = Counter()
        for rec in gold.values():
            if isinstance(rec, dict) and label_key in rec:
                counts[str(rec[label_key])] += 1
        return dict(counts)
    return {"n_items": len(gold)}


def _flatten_metrics(task_dir: str, score_file: Path, data: dict[str, Any]) -> list[dict]:
    rows: list[dict] = []
    base = {
        "task": task_dir.replace("phar_", ""),
        "score_file": score_file.name,
    }
    if "n" in data:
        for metric in ("accuracy", "macro_f1", "mean_score", "avg_score"):
            if metric in data:
                rows.append({**base, "metric_name": metric, "value": data[metric], "n": data.get("n")})
        if "per_class" in data and isinstance(data["per_class"], dict):
            for cls, stats in data["per_class"].items():
                if isinstance(stats, dict) and "f1" in stats:
                    rows.append(
                        {
                            **base,
                            "metric_name": f"f1_{cls}",
                            "value": stats["f1"],
                            "n": data.get("n"),
                        }
                    )
    elif isinstance(data, dict):
        for model_key, metrics in data.items():
            if not isinstance(metrics, dict):
                continue
            for metric, val in metrics.items():
                if metric in ("per_class",):
                    continue
                if isinstance(val, (int, float)):
                    rows.append(
                        {
                            **base,
                            "metric_name": metric,
                            "value": val,
                            "n": metrics.get("n"),
                            "model_key": model_key,
                        }
                    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-dir", type=Path, default=PHAR_ROOT / "scores")
    ap.add_argument("--out-json", type=Path, default=PHAR_ROOT / "scores" / "paper_results.json")
    ap.add_argument("--out-csv", type=Path, default=PHAR_ROOT / "scores" / "paper_results.csv")
    ap.add_argument("--eval-mode", default="agent")
    ap.add_argument("--model", default="")
    ap.add_argument(
        "--notes-json",
        type=Path,
        default=None,
        help="Merge this JSON object into paper_results.json top-level (e.g. per_task_eval_mode)",
    )
    args = ap.parse_args()

    scores_root = args.scores_dir if args.scores_dir.is_absolute() else PHAR_ROOT / args.scores_dir
    paper: dict[str, Any] = {
        "eval_mode": args.eval_mode,
        "model": args.model,
        "metrics": [],
        "gold_distributions": {},
    }
    if args.notes_json and args.notes_json.is_file():
        extra = json.loads(args.notes_json.read_text(encoding="utf-8"))
        if isinstance(extra, dict):
            for k, v in extra.items():
                paper[k] = v
    flat_rows: list[dict[str, Any]] = []

    for task_dir, (gold_rel, label_key) in TASK_GOLD.items():
        gold_path = PHAR_ROOT / gold_rel
        paper["gold_distributions"][task_dir] = _gold_label_dist(gold_path, label_key)

        task_scores = scores_root / task_dir
        if not task_scores.is_dir():
            continue
        for score_file in sorted(task_scores.glob("*.json")):
            if score_file.name in ("paper_results.json",):
                continue
            try:
                data = json.loads(score_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            entry = {
                "task": task_dir,
                "score_file": score_file.name,
                "data": data,
            }
            paper["metrics"].append(entry)
            for row in _flatten_metrics(task_dir, score_file, data):
                row["eval_mode"] = args.eval_mode
                row["model"] = args.model
                flat_rows.append(row)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(paper, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = ["task", "metric_name", "value", "n", "eval_mode", "model", "score_file", "model_key"]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in flat_rows:
            w.writerow(row)

    print(f"Wrote {args.out_json} and {args.out_csv} ({len(flat_rows)} metric rows)")


if __name__ == "__main__":
    main()
