"""Shared caps for agent tools that return post text (context budget)."""

from __future__ import annotations

import os

# data_folder: max posts per call (end_idx - start_idx)
DATA_FOLDER_MAX_SLICE = int(os.environ.get("PHAR_DATA_FOLDER_MAX_SLICE", "12"))
# data_folder: max posts returned across all calls in one agent query
DATA_FOLDER_MAX_TOTAL_READ = int(os.environ.get("PHAR_DATA_FOLDER_MAX_TOTAL_READ", "36"))

# post_retrieve
POST_RETRIEVE_MAX_TOPK = int(os.environ.get("PHAR_POST_RETRIEVE_MAX_TOPK", "8"))
POST_RETRIEVE_FIELD_MAX_CHARS = int(os.environ.get("PHAR_POST_RETRIEVE_FIELD_MAX_CHARS", "800"))


def truncate_post_field(value, max_chars: int = POST_RETRIEVE_FIELD_MAX_CHARS) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
