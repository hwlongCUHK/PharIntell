"""Load /补全 post_search 文件夹对应的帖子向量（与 build_post_emb 布局一致）。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from phar_embedding import get_sentence_transformer

logger = logging.getLogger(__name__)

_EMB_DIR = Path(__file__).resolve().parent / "database" / "emb_data"
_TOPIC_PATH = _EMB_DIR / "topic_data.npy"


def post_text_for_embedding(post: dict) -> str:
    title = post.get("标题") or post.get("title") or ""
    body = post.get("内容") or post.get("content") or ""
    if not isinstance(title, str):
        title = str(title)
    if not isinstance(body, str):
        body = str(body)
    return f"{title.strip()} {body.strip()}".strip() or " "


def emb_paths_for_folder(folder_name: str) -> tuple[Path, Path, str]:
    from tools.folder_name_resolve import safe_folder_filename

    safe = safe_folder_filename(folder_name)
    parts = folder_name.split("_", 2)
    date = parts[1].split(" ")[0] if len(parts) == 3 else ""
    corpus = parts[0] if parts else ""
    p_pharm = _EMB_DIR / f"{safe}.npy"
    p_legacy = _EMB_DIR / f"{date}_{corpus}.npy" if date and corpus else Path()
    return p_pharm, p_legacy, safe


def load_folder_emb_map(folder_name: str, emb_base: dict | None = None) -> dict[str, Any]:
    """uid -> vector；优先 topic_data.npy 内嵌，再 pharm safe .npy，再 legacy。"""
    if emb_base and folder_name in emb_base:
        raw = emb_base[folder_name]
        if isinstance(raw, dict):
            return dict(raw)

    p_pharm, p_legacy, _ = emb_paths_for_folder(folder_name)
    if p_pharm.is_file():
        return dict(np.load(p_pharm, allow_pickle=True).item())
    if p_legacy.is_file():
        return dict(np.load(p_legacy, allow_pickle=True).item())
    return {}


def _encode_missing(
    missing_ids: list[str],
    posts_by_id: dict[str, dict],
    model,
) -> dict[str, np.ndarray]:
    if not missing_ids:
        return {}
    texts = [post_text_for_embedding(posts_by_id[uid]) for uid in missing_ids]
    batch_size = max(1, int(os.environ.get("PHAR_EMBED_BATCH", "8")))
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(texts), batch_size):
        chunk_ids = missing_ids[i : i + batch_size]
        chunk_texts = texts[i : i + batch_size]
        vecs = model.encode(
            chunk_texts,
            prompt_name="document",
            batch_size=min(batch_size, len(chunk_texts)),
            show_progress_bar=False,
        )
        for uid, vec in zip(chunk_ids, vecs):
            out[uid] = np.asarray(vec, dtype=np.float32)
    return out


def maybe_persist_emb_map(folder_name: str, posts_emb: dict[str, Any]) -> None:
    if os.environ.get("PHAR_SKIP_EMB_PERSIST", "").strip().lower() in ("1", "true", "yes"):
        return
    p_pharm, _, _ = emb_paths_for_folder(folder_name)
    try:
        p_pharm.parent.mkdir(parents=True, exist_ok=True)
        np.save(p_pharm, posts_emb)
    except Exception as ex:
        logger.warning("无法写回向量文件 %s: %s", p_pharm, ex)


def ensure_embeddings_for_posts(
    post_ids: list[str],
    posts: list[dict],
    posts_emb: dict[str, Any],
    *,
    model=None,
    persist_folder_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    保证 post_ids 中每条在 posts_emb 里都有向量；缺失则现场 encode。
    返回 (更新后的 posts_emb, 给 agent 的前缀说明)。
    """
    posts_by_id = {str(p["unique_id"]): p for p in posts}
    missing = [uid for uid in post_ids if uid not in posts_emb]
    if not missing:
        return posts_emb, ""

    model = model or get_sentence_transformer()
    added = _encode_missing(missing, posts_by_id, model)
    posts_emb.update(added)

    note = (
        f"【向量补全】{len(missing)} 条帖子不在既有 .npy 中（常见于 post_search 换 keywords 后子集变化），"
        f"已现场编码并参与检索/聚类。"
    )
    if persist_folder_name:
        maybe_persist_emb_map(persist_folder_name, posts_emb)
        note += f" 已尝试写回 database/emb_data/{emb_paths_for_folder(persist_folder_name)[2]}.npy。"
    return posts_emb, note + "\n\n"


def stack_embeddings_for_posts(
    posts: list[dict],
    posts_emb: dict[str, Any],
    *,
    model=None,
    persist_folder_name: str | None = None,
) -> tuple[list[dict], np.ndarray, str]:
    """按 posts 顺序取向量矩阵；缺失 uid 会先 ensure。"""
    ids = [str(p["unique_id"]) for p in posts]
    posts_emb, note = ensure_embeddings_for_posts(
        ids,
        posts,
        posts_emb,
        model=model,
        persist_folder_name=persist_folder_name,
    )
    mats = [np.asarray(posts_emb[uid], dtype=np.float32) for uid in ids]
    return posts, np.stack(mats, axis=0), note
