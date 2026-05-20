#!/usr/bin/env python3
"""Build per-post embedding dict (.npy) for topic_clustering / post_retrieve (Phar folder layout).

Normally ``post_search`` writes the same ``database/emb_data/<safe>.npy`` automatically on first
use. Use this script to (re)build offline, change model/device, or repair a corrupted file.

Device follows ``PHAR_EMBEDDING_DEVICE`` / ``config.embedding_model_device`` (typically ``cuda`` when
available). Progress: ``PHAR_EMBED_SHOW_PROGRESS`` defaults to ``1`` (tqdm bar); set to ``0`` to disable.
Batch: ``PHAR_EMBED_BATCH`` defaults to **8**; encodes in chunks. On CUDA OOM (including
``RuntimeError: CUDA error: out of memory``), batch is halved and the current chunk retried until 1.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phar_embedding import get_sentence_transformer  # noqa: E402
from phar_folder_paths import safe_folder_filename  # noqa: E402


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _cuda_clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def encode_texts_chunked(
    model,
    texts: list[str],
    *,
    initial_batch_size: int,
    show_progress_bar: bool,
) -> list:
    """Encode all texts in micro-batches; halve batch on CUDA OOM and retry the same offset."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    n = len(texts)
    if n == 0:
        return []

    out: list = []
    batch_size = max(1, initial_batch_size)
    i = 0
    pbar = tqdm(total=n, desc="encode", unit="post") if show_progress_bar and tqdm else None

    while i < n:
        chunk_bs = min(batch_size, n - i)
        chunk = texts[i : i + chunk_bs]
        try:
            part = model.encode(
                chunk,
                prompt_name="document",
                batch_size=chunk_bs,
                show_progress_bar=False,
            )
            out.extend(list(part))
            i += chunk_bs
            if pbar is not None:
                pbar.update(chunk_bs)
        except BaseException as ex:
            if not _is_cuda_oom(ex):
                raise
            if batch_size <= 1:
                raise
            _cuda_clear()
            batch_size = max(1, batch_size // 2)
            print(
                f"[build_post_emb] CUDA OOM at offset {i}/{n}, "
                f"retry chunk with batch_size={batch_size}",
                flush=True,
            )

    if pbar is not None:
        pbar.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="JSONL with internal keys (unique_id, title, content, ...)",
    )
    parser.add_argument(
        "--folder-name",
        type=str,
        required=True,
        help="Exact post_search folder name, e.g. brand_2025-03-01 00:00:00_2025-05-01 23:59:59",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("database/emb_data"),
        help="Output directory",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{safe_folder_filename(args.folder_name)}.npy"

    model = get_sentence_transformer()
    try:
        dev_s = str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        dev_s = "unknown"
    print(f"[build_post_emb] device={dev_s}", flush=True)

    posts = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            posts.append(json.loads(line))

    texts = [
        f"{p.get('title', '')} {p.get('content', '')}".strip() or " " for p in posts
    ]
    uids = [str(p["unique_id"]) for p in posts]
    print(f"[build_post_emb] {len(texts)} posts", flush=True)

    batch_size = max(1, int(os.environ.get("PHAR_EMBED_BATCH", "8")))
    if len(texts) > 3000:
        batch_size = min(batch_size, 4)
    show_bar = os.environ.get("PHAR_EMBED_SHOW_PROGRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    all_rows = encode_texts_chunked(
        model,
        texts,
        initial_batch_size=batch_size,
        show_progress_bar=show_bar,
    )

    emb_map = {}
    for uid, row in zip(uids, all_rows):
        emb_map[uid] = np.asarray(row, dtype=np.float32)

    np.save(out_path, emb_map)
    print(f"Saved {len(emb_map)} vectors to {out_path}")


if __name__ == "__main__":
    main()
