"""PharmXHS post search: corpus + time window + optional keywords / drug filter."""
from __future__ import annotations

import json
import os
from pathlib import Path

import json5
import numpy as np
import torch

from qwen_agent.log import logger
from qwen_agent.tools.base import BaseTool, register_tool
from config import post_key2entry
from post_window_filter import _list_raw_jsonl_files, iter_raw_posts_for_window
from .folder_name_resolve import safe_folder_filename


@register_tool("post_search")
class SearchPosts(BaseTool):
    description = (
        "小红书贴文搜索。输入语料库 corpus（如 brand / nonbrand）、开始时间、结束时间；"
        "可选 keywords（空格分隔多个关键词，帖子标题或正文中需至少匹配其一）、"
        "drug_filter（可选，匹配药品标签 drug_label 子串）。"
        "符合条件的帖子写入数据文件夹，文件夹名为 '{corpus}_{start_time}_{end_time}'。"
        "首次检索且磁盘上尚无同名向量文件时，会自动写入 database/emb_data 下与 post_retrieve/topic_clustering 一致的 .npy（已存在则跳过）。"
        "原始 JSONL 放在 database/raw_data/ 下，文件名形如 '{corpus}_*.jsonl'。"
        "若需关闭自动生成，可设置环境变量 PHAR_SKIP_AUTO_POST_EMB=1。"
    )
    parameters = [
        {
            "name": "corpus",
            "type": "string",
            "description": "语料库标识，如 brand、nonbrand；需与 raw_data 下 jsonl 文件名前缀一致",
            "required": True,
        },
        {
            "name": "start_time",
            "type": "string",
            "description": "开始时刻，格式 %Y-%m-%d %H:%M:%S",
            "required": True,
        },
        {
            "name": "end_time",
            "type": "string",
            "description": "结束时刻，格式 %Y-%m-%d %H:%M:%S（区间上界为开区间，与原版一致）",
            "required": True,
        },
        {
            "name": "keywords",
            "type": "string",
            "description": "可选；空格分隔的关键词，至少匹配一个则保留",
            "required": False,
        },
        {
            "name": "drug_filter",
            "type": "string",
            "description": "可选；若帖子含 drug_label 字段，则须包含该子串",
            "required": False,
        },
    ]

    @staticmethod
    def _maybe_write_post_embeddings(folder_name: str, posts: list, posts_idx: list) -> bool:
        """Write database/emb_data/<safe>.npy if missing (same layout as scripts/build_post_emb.py)."""
        if os.environ.get("PHAR_SKIP_AUTO_POST_EMB", "").strip().lower() in ("1", "true", "yes"):
            return False
        if not posts or len(posts) != len(posts_idx):
            return False
        emb_dir = Path(__file__).resolve().parent.parent / "database" / "emb_data"
        emb_dir.mkdir(parents=True, exist_ok=True)
        safe = safe_folder_filename(folder_name)
        p_pharm = emb_dir / f"{safe}.npy"
        if p_pharm.is_file():
            return False
        from phar_embedding import get_sentence_transformer

        model = get_sentence_transformer()
        texts: list[str] = []
        for p in posts:
            title = p.get("标题") or ""
            body = p.get("内容") or ""
            if not isinstance(title, str):
                title = str(title)
            if not isinstance(body, str):
                body = str(body)
            texts.append(f"{title.strip()} {body.strip()}".strip() or " ")
        batch_size = max(1, int(os.environ.get("PHAR_EMBED_BATCH", "32")))
        embeddings = None
        while batch_size >= 1:
            try:
                embeddings = model.encode(
                    texts,
                    prompt_name="document",
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size <= 1:
                    logger.warning(
                        "post_search 自动编码 CUDA OOM（batch_size=1），放弃写入向量文件；"
                        "可改用 CPU / 预跑 scripts/preflight_post_emb_for_benchmark_queries.py 或 build_post_emb.py"
                    )
                    return False
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "post_search 自动编码 CUDA OOM，将 batch_size 降为 %s 后重试", batch_size
                )
        if embeddings is None:
            return False
        emb_map: dict[str, np.ndarray] = {}
        for uid, row_vec in zip(posts_idx, embeddings):
            emb_map[str(uid)] = np.asarray(row_vec, dtype=np.float32)
        np.save(p_pharm, emb_map)
        return True

    def call(self, params: str, data_folder, **kwargs):
        params = json5.loads(params)
        corpus = params["corpus"]
        start_time = params["start_time"]
        end_time = params["end_time"]
        keywords_raw = (params.get("keywords") or "").strip()
        drug_filter = (params.get("drug_filter") or "").strip()

        raw_dir = os.path.join(".", "database", "raw_data")
        files = _list_raw_jsonl_files(corpus, raw_dir)
        if not files:
            return (
                f"未找到语料文件：在 {raw_dir} 下期望 {corpus}_*.jsonl 或 {corpus}.jsonl。"
            )

        filtered_posts = []
        filtered_posts_idx = []

        for post in iter_raw_posts_for_window(
            corpus, start_time, end_time, keywords_raw, drug_filter, raw_dir=raw_dir
        ):
            succinct_post = {
                post_key2entry[k]: post[k]
                for k in post_key2entry
                if k in post
            }
            filtered_posts.append(succinct_post)
            filtered_posts_idx.append(succinct_post["unique_id"])

        folder_name = f"{corpus}_{start_time}_{end_time}"
        data_folder.data_folders[folder_name] = filtered_posts
        data_folder.data_folders[f"{folder_name}_idx"] = filtered_posts_idx
        data_folder.show_funcs[folder_name] = SearchPosts.show
        cluster_name = f"{folder_name}_cluster"
        if not filtered_posts:
            if hasattr(data_folder, "mark_folder_empty"):
                data_folder.mark_folder_empty(folder_name)
                data_folder.mark_folder_empty(cluster_name)
            return (
                f"0 条符合条件的帖子已存储在数据文件夹 '{folder_name}' 中。\n"
                f'  "{folder_name}"\n\n'
                "【停止】当前检索结果为空。请勿调用 topic_clustering、post_retrieve 或 data_folder；"
                "请放宽 keywords、调整时间窗后重新 post_search，"
                "或直接按任务要求输出空 JSON（如 top_concerns/symptom_clusters 等为 []）。\n"
            )

        if hasattr(data_folder, "clear_folder_empty"):
            data_folder.clear_folder_empty(folder_name)
            data_folder.clear_folder_empty(cluster_name)

        emb_dir = Path(__file__).resolve().parent.parent / "database" / "emb_data"
        safe = safe_folder_filename(folder_name)
        p_pharm = emb_dir / f"{safe}.npy"

        emb_note = ""
        try:
            if SearchPosts._maybe_write_post_embeddings(folder_name, filtered_posts, filtered_posts_idx):
                emb_note = f"已自动生成帖子向量文件（{safe_folder_filename(folder_name)}.npy），post_retrieve / topic_clustering 可直接使用。\n"
        except Exception as ex:
            logger.warning(
                "post_search 自动生成帖子向量失败（可稍后手工运行 scripts/build_post_emb.py）: %s",
                ex,
            )

        emb_missing = ""
        if filtered_posts and not p_pharm.is_file():
            emb_missing = (
                "\n【重要】当前文件夹内帖子已就绪，但未生成或未找到对应向量文件 "
                f"`database/emb_data/{p_pharm.name}`，因此 **topic_clustering** 与 **post_retrieve** 将无法使用，"
                "直至编码成功。可选处理：在空闲 GPU 或 CPU 上执行 "
                f'`python scripts/build_post_emb.py --folder-name "{folder_name}"`；'
                "或设置 `PHAR_EMBEDDING_DEVICE=cpu` / 减小 `PHAR_EMBEDDING_MODEL` 占用后重试自动编码；"
                "或设置 `PHAR_SKIP_AUTO_POST_EMB=1` 并离线生成 `.npy`。"
                "仍可用 **data_folder** 小批量阅读（单次≤12 条，优先读 _cluster 文件夹）。\n"
            )

        return (
            f"{len(filtered_posts)} 条符合条件的帖子已存储在数据文件夹 '{folder_name}' 中。\n"
            f"后续 topic_clustering / post_retrieve 时，参数 folder_name 必须与下面引号内字符串完全一致"
            f"（含日期里的空格与冒号；不要改成下划线版文件名，也不要缩写）：\n"
            f'  "{folder_name}"\n'
            f"下一步必须 topic_clustering，再用 data_folder 读 "
            f'"{folder_name}_cluster"（簇编号，非帖序号）。'
            f"可选 post_retrieve（topk≤8）仅作补充。"
            f"最终 JSON 条数按话题异质化 1–min(K,8) 条，勿固定凑满 5 条。\n"
            f"{emb_note}"
            f"{emb_missing}"
        )

    @staticmethod
    def show(posts: list, start_idx: int, end_idx: int) -> str:
        if not posts:
            return "没有找到符合条件的帖子。"

        if start_idx < 0:
            raise ValueError("起始索引不能小于0。")
        elif end_idx > len(posts):
            raise ValueError(f"结束索引不能超过帖子列表的长度{len(posts)}。")
        elif start_idx >= end_idx:
            raise ValueError("起始索引必须小于结束索引。")

        result = ""
        for i in range(start_idx, end_idx):
            post = posts[i]
            result += f"帖子 {i + 1}:\n"
            for key, value in post.items():
                if key == "unique_id":
                    continue
                text = value if isinstance(value, str) else str(value)
                result += f"***{key}***: {text[:512]}\n"
            result += "\n"

        return result.strip()
