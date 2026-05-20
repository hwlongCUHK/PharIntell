#!/usr/bin/env python3
"""Convert PharmXHS CSV exports to PharIntell raw_data JSONL (internal keys for post_search)."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import uuid
from pathlib import Path


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text).strip())


def row_to_post(row: dict) -> dict:
    return {
        "unique_id": str(uuid.uuid4()),
        "content": strip_html(row.get("news_content", "")),
        "title": (row.get("news_title") or "").strip(),
        "post_publish_time": (row.get("news_posttime") or "").strip(),
        "ocr": "",
        "platform_name": "小红书",
        "nickname": (row.get("media_organization") or "个人").strip() or "个人",
        "public_location": "",
        "like_count": int(float(row.get("news_like_count") or 0)),
        "drug_label": (row.get("drug_label") or "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV path (e.g. ../data/drug_classified.csv)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Corpus prefix for output filename, e.g. brand / nonbrand",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("database/raw_data"),
        help="Output directory (default: database/raw_data under PharIntell)",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="",
        help="Optional output basename without .jsonl (default: {corpus}_export)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_name or f"{args.corpus}_export"
    out_path = out_dir / f"{base}.jsonl"

    with args.csv.open("r", encoding="utf-8", newline="") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        n = 0
        for row in reader:
            post = row_to_post(row)
            if not post["post_publish_time"]:
                continue
            fout.write(json.dumps(post, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} posts to {out_path}")


if __name__ == "__main__":
    main()
