#!/usr/bin/env python3
"""Encode knowledge_base.json with the Phar embedding model and save knowledge_base.npy."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    embedding_model_device,
    embedding_model_path,
    knowledge_emb_path,
    knowledge_path,
)


def _effective_batch_size(n: int, requested: int, *, min_steps: int = 8) -> int:
    """Shrink batch so tqdm has multiple steps (meaningless if only 0/1 for tiny corpora)."""
    if n <= 0:
        return max(1, requested)
    cap_steps = min(n, max(min_steps, min(32, n)))
    bs = max(1, requested)
    while bs > 1 and math.ceil(n / bs) < cap_steps:
        bs = max(1, bs // 2)
    return bs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding (smaller if GPU OOM; auto-shrunk when doc count is small so tqdm has multiple steps)",
    )
    args = parser.parse_args()

    with open(knowledge_path, encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d.get("content", "") for d in docs]
    n = len(texts)

    print(f"Loading SentenceTransformer from: {embedding_model_path}", flush=True)
    model = SentenceTransformer(
        embedding_model_path,
        device=embedding_model_device,
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    if n == 0:
        emb = np.zeros((0, int(model.get_sentence_embedding_dimension())), dtype=np.float32)
    else:
        bs = _effective_batch_size(n, args.batch_size)
        n_batches = math.ceil(n / bs)
        if bs != args.batch_size:
            print(
                f"Adjusted batch_size {args.batch_size} -> {bs} "
                f"({n_batches} steps for {n} docs; avoids a useless 0/1 bar)",
                flush=True,
            )
        parts: list = []
        for start in tqdm(
            range(0, n, bs),
            desc="Encoding knowledge",
            unit="batch",
            total=n_batches,
        ):
            chunk = texts[start : start + bs]
            parts.append(
                model.encode(
                    chunk,
                    prompt_name="document",
                    show_progress_bar=False,
                )
            )
        emb = np.vstack(parts)

    out_path = Path(knowledge_emb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    print(f"Saved {emb.shape} embeddings to {out_path}", flush=True)


if __name__ == "__main__":
    main()
