import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import json5
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from typing import Optional

from qwen_agent.tools.base import BaseTool, register_tool
from config import embedding_model_path, embedding_model_device, knowledge_path, knowledge_emb_path


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "alloc" in msg)


def _canonical_embed_device(dev: str) -> str:
    """Normalize for comparison / ordering (cuda -> cuda:0)."""
    if not dev:
        return "cpu"
    d = dev.strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "cuda":
        return "cuda:0"
    return dev.strip()


def _primary_cuda_index(primary: str) -> Optional[int]:
    """Index for config primary device, or None if not CUDA."""
    pl = primary.strip().lower()
    if pl == "cpu":
        return None
    if pl == "cuda":
        return 0
    if pl.startswith("cuda:"):
        try:
            return int(primary.split(":", 1)[1].strip())
        except ValueError:
            return 0
    return None


def _embedding_device_try_order() -> list[str]:
    """
    Devices to try for SentenceTransformer: primary first, then other CUDA
    GPUs ordered by free memory (descending), then CPU.
    """
    primary = embedding_model_device.strip()
    pl = primary.lower()
    if pl == "cpu":
        return ["cpu"]
    if not torch.cuda.is_available():
        return ["cpu"]
    n = torch.cuda.device_count()
    if n <= 0:
        return ["cpu"]

    pi = _primary_cuda_index(primary)
    order: list[str] = []
    if pi is not None and 0 <= pi < n:
        order.append(f"cuda:{pi}")
    elif pl == "cuda":
        order.append("cuda:0")
    elif pl.startswith("cuda:"):
        order.append("cuda:0")
    else:
        # MPS / non-CUDA: no multi-GPU fallback chain here
        return [primary, "cpu"]

    seen = {_canonical_embed_device(d) for d in order}
    rest_idx = [i for i in range(n) if f"cuda:{i}" not in seen]

    def _free_bytes(idx: int) -> int:
        try:
            free, _ = torch.cuda.mem_get_info(idx)
            return int(free)
        except Exception:
            return 0

    rest_idx.sort(key=_free_bytes, reverse=True)
    order.extend(f"cuda:{i}" for i in rest_idx)
    order.append("cpu")
    return order


def _devices_after(failed: str, full_order: list[str]) -> list[str]:
    """Skip through full_order until after `failed` (canonical match)."""
    fc = _canonical_embed_device(failed)
    for i, d in enumerate(full_order):
        if _canonical_embed_device(d) == fc:
            return full_order[i + 1 :]
    return [d for d in full_order if _canonical_embed_device(d) != fc]


@register_tool("knowledge_retrieve")
class RetrieveKnowledge(BaseTool):
    description = "Service for social media knowledge retrieval. Enter the retrieval query and topk, retrieve the knowledge documents matching the query semantics from the knowledge base, and return the topk most relevant knowledge documents."
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "retrieval query",
            "required": True,
        },
        {
            "name": "topk",
            "type": "int",
            "description": "number of returned knowledge documents",
            "required": True,
        },
    ]

    _kb_ready = False
    model = None
    documents = None
    document_embeddings = None
    _embed_device: str = ""

    @classmethod
    def _reset_kb_state(cls) -> None:
        cls._kb_ready = False
        cls.model = None
        cls.documents = None
        cls.document_embeddings = None
        cls._embed_device = ""
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def _load_st_model(cls, device: str) -> None:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        cls.model = SentenceTransformer(
            embedding_model_path,
            device=device,
            model_kwargs={"torch_dtype": dtype},
            tokenizer_kwargs={"padding_side": "left"},
        )
        cls._embed_device = device

    @classmethod
    def _ensure_knowledge(cls):
        if cls._kb_ready:
            return
        if not os.path.isfile(knowledge_path):
            raise FileNotFoundError(
                f"知识库 JSON 不存在: {knowledge_path}。请放置说明书片段或设置 PHAR_KNOWLEDGE_JSON。"
            )
        if not os.path.isfile(knowledge_emb_path):
            raise FileNotFoundError(
                f"知识库向量不存在: {knowledge_emb_path}。请在 PharIntell 目录下运行 "
                f"`python scripts/build_knowledge_emb.py` 生成，或设置 PHAR_KNOWLEDGE_NPY。"
            )
        for dev in _embedding_device_try_order():
            try:
                cls._load_st_model(dev)
                break
            except Exception as e:
                if dev != "cpu" and _is_cuda_oom(e):
                    cls._reset_kb_state()
                    continue
                raise
        with open(knowledge_path, encoding="utf-8") as f:
            cls.documents = json.load(f)
        cls.document_embeddings = np.load(knowledge_emb_path)
        if len(cls.documents) != cls.document_embeddings.shape[0]:
            raise ValueError(
                f"知识库条数不一致: JSON {len(cls.documents)} 条 vs "
                f"{knowledge_emb_path} 行数 {cls.document_embeddings.shape[0]}。"
                "请重新运行 scripts/build_knowledge_emb.py。"
            )
        model_dim = int(cls.model.get_sentence_embedding_dimension())
        emb_dim = int(cls.document_embeddings.shape[-1])
        if model_dim != emb_dim:
            raise ValueError(
                f"知识库向量维度 {emb_dim} 与当前嵌入模型维度 {model_dim} 不一致。\n"
                f"  模型: {embedding_model_path}\n"
                f"  向量: {knowledge_emb_path}\n"
                "请用同一模型重建: cd PharIntell && PYTHONPATH=. python scripts/build_knowledge_emb.py\n"
                "或设置 PHAR_EMBEDDING_MODEL 为构建该 .npy 时使用的模型快照路径。"
            )
        cls._kb_ready = True

    def call(self, params: str, **kwargs):
        self._ensure_knowledge()
        # json5.loads breaks non-BMP chars (e.g. emoji) into lone surrogates and
        # breaks HuggingFace fast tokenizers (TextEncodeInput). Prefer stdlib json.
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = json5.loads(params)
        query, topk = params["query"], params["topk"]
        if isinstance(query, (list, tuple)):
            query = " ".join(str(x) for x in query if str(x).strip()).strip()
        else:
            query = str(query).strip() if query is not None else ""
        # Repair any lone surrogates (e.g. after json5) so tokenizers accept input.
        try:
            query = query.encode("utf-16", "surrogatepass").decode("utf-16")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        if not query:
            query = "药品说明书"
        query = query.replace("\x00", "")[:8000]

        if topk <= 0:
            raise ValueError("The value of topk should be positive integer!")

        def _encode_and_rank() -> list:
            query_embedding = self.model.encode([query], prompt_name="query")
            similarity = self.model.similarity(query_embedding, self.document_embeddings)
            similarity = similarity.squeeze(0)
            ranked_indices = torch.argsort(similarity, descending=True)[:topk]
            return [self.documents[int(i)] for i in ranked_indices]

        try:
            retrieved_posts = _encode_and_rank()
        except Exception as e:
            if self._embed_device == "cpu" or not _is_cuda_oom(e):
                raise
            failed = self._embed_device
            self._reset_kb_state()
            last_err: BaseException = e
            for dev in _devices_after(failed, _embedding_device_try_order()):
                try:
                    self._load_st_model(dev)
                    with open(knowledge_path, encoding="utf-8") as f:
                        self.documents = json.load(f)
                    self.document_embeddings = np.load(knowledge_emb_path)
                    model_dim = int(self.model.get_sentence_embedding_dimension())
                    emb_dim = int(self.document_embeddings.shape[-1])
                    if model_dim != emb_dim:
                        raise ValueError(
                            f"知识库向量维度 {emb_dim} 与嵌入模型维度 {model_dim} 不一致；"
                            "请运行 scripts/build_knowledge_emb.py 重建 knowledge_base.npy。"
                        )
                    self._kb_ready = True
                    retrieved_posts = _encode_and_rank()
                    break
                except Exception as e2:
                    last_err = e2
                    if dev != "cpu" and _is_cuda_oom(e2):
                        self._reset_kb_state()
                        continue
                    raise
            else:
                raise last_err
        return self.show(retrieved_posts)

    @staticmethod
    def show(docs):
        result = "The following are results from the knowledge base:\n\n"

        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            link = doc.get("link", "")
            result += f"Document {i+1}:\n***Content***: {content}\n***Source URL***: {link}\n\n"

        return result
