"""Shared helpers for PharIntell eval_scripts."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

PHAR_ROOT = Path(__file__).resolve().parents[1]


def strip_think_tags(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"\[THINK\].*?\[ANSWER\]", "", text, flags=re.DOTALL)
    return text.strip()


def strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t, count=1)
    return t.strip()


def load_eval_settings(setting_path: str | Path) -> dict[str, Any]:
    p = Path(setting_path)
    if not p.is_absolute():
        p = PHAR_ROOT / p
    if not p.is_file():
        example = PHAR_ROOT / "eval_scripts" / "settings.json.example"
        raise FileNotFoundError(
            f"Eval settings not found: {p}. Copy {example} to eval_scripts/settings.json "
            "or set PHAR_EVAL_MODEL / PHAR_EVAL_BASE_URL / PHAR_EVAL_API_KEY."
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {
        "model": os.environ.get("PHAR_EVAL_MODEL", data.get("model", "")),
        "base_url": os.environ.get("PHAR_EVAL_BASE_URL", data.get("base_url", "")),
        "api_key": os.environ.get("PHAR_EVAL_API_KEY", data.get("api_key", "")),
    }
    return out


def first_json_object(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from model output."""
    cleaned = strip_think_tags(strip_markdown_fences(text))
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def resolve_under_phar(*parts: str) -> Path:
    return PHAR_ROOT.joinpath(*parts)


def list_result_jsons(result_dir: Path, prefix: str) -> list[Path]:
    if not result_dir.is_dir():
        return []
    out: list[Path] = []
    for name in sorted(result_dir.iterdir()):
        if not name.is_file() or not name.suffix == ".json":
            continue
        if name.name.startswith(prefix + "_"):
            out.append(name)
    return out
