"""Raw JSONL post iteration by corpus + time window + optional keywords / drug filter.

Kept at package root (not under ``tools/``) so scripts like ``preflight_post_emb_for_benchmark_queries.py``
can import without executing ``tools/__init__.py`` (which registers tools and pulls heavy ML deps).
"""
from __future__ import annotations

import glob
import json
import os
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Any


def _list_raw_jsonl_files(corpus: str, raw_dir: str | None = None) -> list[str]:
    base = raw_dir or os.path.join(".", "database", "raw_data")
    pattern = os.path.join(base, f"{corpus}_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        single = os.path.join(base, f"{corpus}.jsonl")
        if os.path.isfile(single):
            files = [single]
    return files


def iter_raw_posts_for_window(
    corpus: str,
    start_time: str,
    end_time: str,
    keywords_raw: str = "",
    drug_filter: str = "",
    *,
    raw_dir: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw post dicts from JSONL matching the same rules as post_search (internal keys)."""
    files = _list_raw_jsonl_files(corpus, raw_dir)
    if not files:
        return
    date_form = "%Y-%m-%d %H:%M:%S"
    t0 = datetime.strptime(start_time, date_form)
    t1 = datetime.strptime(end_time, date_form)
    kws = [k for k in (keywords_raw or "").split() if k]
    df = (drug_filter or "").strip()
    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                post = json.loads(line)
                if "unique_id" not in post:
                    post["unique_id"] = str(uuid.uuid4())
                pt = post.get("post_publish_time")
                if not pt:
                    continue
                try:
                    pub = datetime.strptime(pt, date_form)
                except ValueError:
                    continue
                if not (t0 <= pub < t1):
                    continue
                if df and df not in post.get("drug_label", ""):
                    continue
                if kws:
                    blob = f"{post.get('title', '')} {post.get('content', '')}"
                    if not any(k in blob for k in kws):
                        continue
                yield post
