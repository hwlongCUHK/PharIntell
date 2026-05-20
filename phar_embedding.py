"""Process-local singleton for the Qwen3 / SentenceTransformer embedding model.

Parallel benchmark runs spawn **separate OS processes** (one per model). Each process
has its own address space: PyTorch CUDA tensors cannot be shared across processes
without a dedicated serving process (RPC/HTTP). This module deduplicates loads **within
one process** (e.g. ``post_retrieve`` + ``topic_clustering``), which avoids holding two
copies of the same weights in VRAM in a single ``run_phar_task.py``.

For many models in parallel on one GPU, set ``PHAR_EMBEDDING_DEVICE=cpu`` (see
``config.embedding_model_device``) so local tools do not contend for CUDA memory.
"""
from __future__ import annotations

import threading
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from config import embedding_model_path, embedding_model_device

_lock = threading.Lock()
_model: Optional[SentenceTransformer] = None


def get_sentence_transformer() -> SentenceTransformer:
    """Return the shared SentenceTransformer for this process (lazy, thread-safe)."""
    global _model
    with _lock:
        if _model is None:
            dev = (embedding_model_device or "cpu").strip().lower()
            dtype = torch.bfloat16 if dev != "cpu" else torch.float32
            try:
                _model = SentenceTransformer(
                    embedding_model_path,
                    device=embedding_model_device,
                    model_kwargs={"dtype": dtype},
                    tokenizer_kwargs={"padding_side": "left"},
                )
            except TypeError:
                _model = SentenceTransformer(
                    embedding_model_path,
                    device=embedding_model_device,
                    model_kwargs={"torch_dtype": dtype},
                    tokenizer_kwargs={"padding_side": "left"},
                )
        return _model
