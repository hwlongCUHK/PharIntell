"""PharIntell runtime configuration: paths, embeddings, post field mapping."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
DATABASE_ROOT = _ROOT / "database"

# Default: Hugging Face Hub id (override with PHAR_EMBEDDING_MODEL=/path/to/local/snapshot).
_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

embedding_model_path = os.environ.get("PHAR_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)


def _default_embedding_device() -> str:
    """Use GPU when available unless PHAR_EMBEDDING_DEVICE is set."""
    override = os.environ.get("PHAR_EMBEDDING_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


embedding_model_device = _default_embedding_device()

knowledge_path = os.environ.get(
    "PHAR_KNOWLEDGE_JSON",
    str(DATABASE_ROOT / "knowledge_data" / "knowledge_base.json"),
)
knowledge_emb_path = os.environ.get(
    "PHAR_KNOWLEDGE_NPY",
    str(DATABASE_ROOT / "emb_data" / "knowledge_base.npy"),
)

# Internal JSONL keys -> keys shown to the agent (aligned with data_folder / topic tools)
post_key2entry = {
    "unique_id": "unique_id",
    "content": "内容",
    "platform_name": "发布平台",
    "title": "标题",
    "post_publish_time": "发布时间",
    "ocr": "OCR",
    "nickname": "发布用户",
    "public_location": "发布地点",
    "like_count": "like_count",
    "drug_label": "药品标签",
}

MAX_LLM_CALL_PER_RUN = int(os.environ.get("PHAR_MAX_LLM_CALL_PER_RUN", "20"))
