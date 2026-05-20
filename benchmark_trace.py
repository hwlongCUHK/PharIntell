"""Append-only JSONL trace for benchmark runs (LLM rounds, tool I/O, query outcomes).

Env:
  PHAR_TRACE_MAX_CHARS — empty or unset: no truncation (full text in previews).
  Set to a positive integer (e.g. 12000) to cap preview fields again.
  PHAR_TRACE_LOG_SYSTEM — 1/true: log full system prompt in input_digest (when trunc enabled, still capped).

Assistant messages may include ``reasoning_content`` (chain-of-thought). The console logger prints it as
``[THINK]`` (see ``qwen_agent/utils/output_beautify.py``). Trace events include ``reasoning_chars`` /
``reasoning_preview`` when present so JSONL aligns with ``run_infer.log`` / tee output.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from qwen_agent.llm.schema import SYSTEM, Message

_TOOL_ERR_MARKERS = (
    "An error occurred when calling tool",
    "ToolServiceError",
    "DocParserError",
)


def trace_max_chars() -> int | None:
    """None means do not truncate trace preview fields."""
    raw = os.environ.get("PHAR_TRACE_MAX_CHARS", "").strip()
    if raw == "":
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n <= 0:
        return None
    return max(500, n)


def trace_log_full_system() -> bool:
    return os.environ.get("PHAR_TRACE_LOG_SYSTEM", "").strip().lower() in ("1", "true", "yes")


def _trunc(s: str, limit: int | None = None) -> str:
    lim = limit if limit is not None else trace_max_chars()
    if lim is None:
        return s or ""
    if not s:
        return ""
    if len(s) <= lim:
        return s
    return s[:lim] + f"\n…[truncated, total_chars={len(s)}]"


def flatten_message_text(msg: Message) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for item in c or []:
        t = getattr(item, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


def flatten_reasoning_text(msg: Message) -> str:
    """Model chain-of-thought; same source as ``[THINK]`` blocks in ``run_infer.log``."""
    c = msg.reasoning_content
    if not c:
        return ""
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for item in c or []:
        t = getattr(item, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


def digest_messages_for_trace(messages: list[Message]) -> list[dict[str, Any]]:
    """Compact representation of conversation sent to the LLM (optional size cap via PHAR_TRACE_MAX_CHARS)."""
    lim = trace_max_chars()
    out: list[dict[str, Any]] = []
    full_sys = trace_log_full_system()
    n_msg = max(4, len(messages))
    for m in messages:
        text = flatten_message_text(m)
        reasoning = flatten_reasoning_text(m)
        fc = m.function_call
        row: dict[str, Any] = {
            "role": m.role,
            "name": m.name,
            "chars": len(text),
        }
        if lim is None:
            row["preview"] = text
            arg_lim: int | None = None
            per_msg = None
        else:
            per_msg = max(800, lim // n_msg)
            if m.role == SYSTEM and not full_sys:
                row["preview"] = _trunc(text, min(400, per_msg))
            else:
                row["preview"] = _trunc(text, per_msg)
            arg_lim = min(per_msg, 4000)
        if reasoning:
            row["reasoning_chars"] = len(reasoning)
            if lim is None:
                row["reasoning_preview"] = reasoning
            else:
                assert per_msg is not None
                row["reasoning_preview"] = _trunc(reasoning, per_msg)
        if fc is not None:
            row["function_call"] = {
                "name": fc.name,
                "arguments": _trunc(fc.arguments or "", arg_lim),
            }
        out.append(row)
    return out


def is_tool_error_result(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    head = result[:1200]
    return any(m in head for m in _TOOL_ERR_MARKERS)


class BenchmarkTraceWriter:
    """One JSONL file per benchmark process; thread-safe enough for single-threaded run_phar_task."""

    def __init__(
        self,
        path: str | Path,
        *,
        task: str,
        model: str,
        eval_mode: str,
    ) -> None:
        self.path = Path(path)
        self.task = task
        self.model = model
        self.eval_mode = eval_mode
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._qid: Optional[str] = None
        self._stats = {"n_llm_rounds": 0, "n_tools": 0, "n_tool_errors": 0}

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> BenchmarkTraceWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        line = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "task": self.task,
            "model": self.model,
            "eval_mode": self.eval_mode,
            **payload,
        }
        self._fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def start_query(self, qid: str, query: str) -> None:
        self._qid = qid
        self._stats = {"n_llm_rounds": 0, "n_tools": 0, "n_tool_errors": 0}
        self._emit(
            "query_start",
            {
                "qid": qid,
                "query_preview": _trunc(query or "", trace_max_chars()),
            },
        )

    def log_llm_round_start(self, round_idx: int, messages: list[Message]) -> None:
        self._stats["n_llm_rounds"] = max(self._stats["n_llm_rounds"], round_idx)
        self._emit(
            "llm_round_start",
            {
                "qid": self._qid,
                "round": round_idx,
                "n_messages": len(messages),
                "input_digest": digest_messages_for_trace(messages),
            },
        )

    def log_llm_output(self, output: list[Message]) -> None:
        dig: list[dict[str, Any]] = []
        for m in output:
            text = flatten_message_text(m)
            reasoning = flatten_reasoning_text(m)
            item: dict[str, Any] = {
                "role": m.role,
                "name": m.name,
                "chars": len(text),
                "preview": _trunc(text, trace_max_chars()),
            }
            if reasoning:
                item["reasoning_chars"] = len(reasoning)
                item["reasoning_preview"] = _trunc(reasoning, trace_max_chars())
            fc = m.function_call
            if fc is not None:
                item["function_call"] = {
                    "name": fc.name,
                    "arguments": _trunc(fc.arguments or "", trace_max_chars()),
                }
            dig.append(item)
        self._emit("llm_output", {"qid": self._qid, "messages": dig})

    def log_tool(
        self,
        tool_name: str,
        tool_args: str | dict,
        result: Any,
    ) -> None:
        self._stats["n_tools"] += 1
        args_s = tool_args if isinstance(tool_args, str) else json.dumps(tool_args, ensure_ascii=False)
        res_s = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        err = is_tool_error_result(res_s)
        if err:
            self._stats["n_tool_errors"] += 1
        self._emit(
            "tool_call",
            {
                "qid": self._qid,
                "tool": tool_name,
                "arguments_preview": _trunc(args_s, trace_max_chars()),
                "result_preview": _trunc(res_s, trace_max_chars()),
                "tool_error": err,
            },
        )

    def log_retry(self, attempt: int, exc: BaseException) -> None:
        self._emit(
            "query_retry",
            {
                "qid": self._qid,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error_preview": _trunc(str(exc), trace_max_chars()),
            },
        )

    def end_query(
        self,
        *,
        status: str,
        final_text: str,
        retries_used: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "qid": self._qid,
            "status": status,
            "retries_used": retries_used,
            "final_chars": len(final_text or ""),
            "final_preview": _trunc(final_text or "", trace_max_chars()),
            "n_llm_rounds": self._stats["n_llm_rounds"],
            "n_tools": self._stats["n_tools"],
            "n_tool_errors": self._stats["n_tool_errors"],
        }
        if extra:
            payload.update(extra)
        self._emit("query_end", payload)


def default_trace_path(output_path: str, task: str, model: str) -> Path:
    safe_model = model.replace("/", "_").replace(" ", "_")
    return Path(output_path).parent / "traces" / f"{task}_{safe_model}.jsonl"
