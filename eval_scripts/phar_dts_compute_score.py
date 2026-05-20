#!/usr/bin/env python3
"""DTS: aggregate mean rubric total per variant scores directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib_phar_eval import PHAR_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["symptom", "treatment", "need"],
        required=True,
    )
    ap.add_argument("--scores_path", type=str, default="")
    args = ap.parse_args()

    root = Path(args.scores_path or f"scores/phar_dts_{args.variant}")
    if not root.is_absolute():
        root = PHAR_ROOT / root

    summary = {}
    for f in sorted(root.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        totals = [v.get("total", 0) for v in data.values() if isinstance(v, dict)]
        if not totals:
            continue
        summary[f.name] = {
            "n": len(totals),
            "mean_total": sum(totals) / len(totals),
            "max_possible": 10,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
