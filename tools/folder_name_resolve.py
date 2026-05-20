"""Resolve disk-safe folder strings to in-memory canonical keys on DataFolder."""
from __future__ import annotations

from typing import Any

from qwen_agent.log import logger

from phar_folder_paths import safe_folder_filename


def resolve_folder_name(data_folder: Any, folder_name: str) -> str:
    """Return canonical key in data_folder.data_folders.

    Models often pass the safe filename form; post_search stores canonical keys
    (with spaces in datetimes). topic_clustering must see canonical form before
    corpus_start_end split.
    """
    if not folder_name or not str(folder_name).strip():
        raise ValueError("folder_name 为空。")
    fn = str(folder_name).strip()
    folders = getattr(data_folder, "data_folders", None)
    if not isinstance(folders, dict):
        raise TypeError("data_folder 缺少 data_folders 字典。")

    if fn in folders:
        return fn

    target_safe = safe_folder_filename(fn)
    candidates = [k for k in folders if safe_folder_filename(k) == target_safe]
    if len(candidates) == 1:
        canonical = candidates[0]
        if canonical != fn:
            logger.info(
                "folder_name 已从 safe/变体形式解析为会话内键: %r -> %r", fn, canonical
            )
        return canonical
    if not candidates:
        raise ValueError(
            f"文件夹'{folder_name}'不存在。请先创建文件夹或检查名称是否正确。"
            f"（若你传的是磁盘向量文件名形式，请改用 post_search 返回中带空格、冒号的 folder_name；"
            f"safe 形式为 {target_safe!r}。）"
        )
    raise ValueError(
        f"文件夹名歧义：与 safe 形式 {target_safe!r} 匹配的键有多个: {candidates!r}"
    )
