import os

import json5
import numpy as np
import torch

from phar_embedding import get_sentence_transformer
from phar_post_emb import load_folder_emb_map, stack_embeddings_for_posts
from phar_tool_limits import (
    POST_RETRIEVE_MAX_TOPK,
    truncate_post_field,
)
from qwen_agent.tools.base import BaseTool, register_tool

from .folder_name_resolve import resolve_folder_name


@register_tool('post_retrieve')
class RetrievePosts(BaseTool):
    description = (
        "社交媒体贴文语义检索：在 post_search 生成的文件夹内按 query 返回最相关帖子。"
        f"topk 不得超过 {POST_RETRIEVE_MAX_TOPK}；优先于 data_folder 大批量读帖。"
    )
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '检索查询',
        'required': True
    },
                  {
        'name': 'folder_name',
        'type': 'string',
        'description': '需要查询的数据文件夹名称',
        'required': True
    },
                  {
        'name': 'topk',
        'type': 'int',
        'description': f'返回条数（1–{POST_RETRIEVE_MAX_TOPK}，超出将被截断）',
        'required': True
    }]
    _topic_path = "./database/emb_data/topic_data.npy"
    emb_base = np.load(_topic_path, allow_pickle=True).item() if os.path.isfile(_topic_path) else {}

    def call(self, params: str, data_folder, **kwargs):
        model = get_sentence_transformer()
        params = json5.loads(params)
        query, folder_name, topk = params['query'], params['folder_name'], params['topk']
        folder_name = resolve_folder_name(data_folder, folder_name)

        if hasattr(data_folder, "folder_has_no_posts") and data_folder.folder_has_no_posts(folder_name):
            return data_folder.EMPTY_FOLDER_REPLY

        documents = data_folder.data_folders[folder_name]
        if not documents:
            return data_folder.EMPTY_FOLDER_REPLY

        posts_emb = load_folder_emb_map(folder_name, self.emb_base)

        documents, document_embeddings, emb_note = stack_embeddings_for_posts(
            documents,
            posts_emb,
            model=model,
            persist_folder_name=folder_name,
        )

        query_embedding = model.encode([query], prompt_name="query")
        similarity = model.similarity(query_embedding, document_embeddings)
        similarity = similarity.squeeze(0)
        n_docs = int(similarity.shape[0])
        if n_docs == 0:
            return emb_note + "没有找到符合条件的帖子。"

        topk_req = max(0, int(topk))
        if topk_req == 0:
            return emb_note + "没有找到符合条件的帖子。"
        capped = min(topk_req, POST_RETRIEVE_MAX_TOPK)
        take = min(capped, n_docs)
        ranked_indices = torch.argsort(similarity, descending=True)[:take]

        retrieved_posts = [documents[int(i)] for i in ranked_indices]
        k = len(retrieved_posts)
        prefix = emb_note
        if topk_req > POST_RETRIEVE_MAX_TOPK:
            prefix += (
                f"请求 topk={topk_req} 超过上限 {POST_RETRIEVE_MAX_TOPK}，已返回 {k} 条"
                f"（文件夹共 {n_docs} 条）。\n\n"
            )
        elif topk_req > k:
            prefix += f"请求 topk={topk_req}，该文件夹内可检索帖子共 {n_docs} 条，已返回 {k} 条。\n\n"
        return prefix + self.show(retrieved_posts, 0, k)

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
                    result += f"***{key}***: {value}\n"
                else:
                    result += f"***{key}***: {truncate_post_field(value)}\n"
            result += "\n"

        return result.strip()
