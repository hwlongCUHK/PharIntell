#!/usr/bin/env python3
"""Run a single PharmXHS task over a JSONL query file (one JSON object per line: id, query)."""
from __future__ import annotations

import os

# Before heavy deps (transformers / optional TensorFlow): reduce TF+XLA stderr noise per subprocess.
for _k, _v in (
    ("TF_CPP_MIN_LOG_LEVEL", "2"),
    ("TF_ENABLE_ONEDNN_OPTS", "0"),
):
    os.environ.setdefault(_k, _v)

import argparse
import importlib
import json
import re
from pathlib import Path
from time import sleep

from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

from agent import SocialMediaAgent
from benchmark_trace import BenchmarkTraceWriter, default_trace_path
from qwen_agent.llm.base import ModelServiceError
from qwen_agent.utils.output_beautify import typewriter_print
from tools.topic_summarization import TopicSummarization

import tools  # noqa: F401  registers post_search, data_folder, ...
import tools.en  # noqa: F401  registers knowledge_retrieve

colorama_init(autoreset=False)


def _trunc_err(exc: BaseException, limit: int = 800) -> str:
    s = f"{type(exc).__name__}: {exc}"
    return s if len(s) <= limit else s[:limit] + "…"


def remove_think_tags(text: str) -> str:
    cleaned = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL
    )
    return cleaned.strip()


def _data_folder_tool(bot: SocialMediaAgent):
    return bot.function_map.get("data_folder")


def _reset_data_folder_read_budget(bot: SocialMediaAgent) -> None:
    df = _data_folder_tool(bot)
    if df is not None and hasattr(df, "reset_read_budget"):
        df.reset_read_budget()


def _reset_data_folder_full(bot: SocialMediaAgent) -> None:
    df = _data_folder_tool(bot)
    if df is not None and hasattr(df, "initialize"):
        df.initialize()


def load_prompt_and_tools(task: str, with_summarize: bool, llm: dict):
    if task == "dcv":
        m = importlib.import_module("tasks.pharm_dcv.prompt")
        zh, fl = m.zh_system, list(m.function_list)
    elif task == "phv_ade":
        m = importlib.import_module("tasks.pharm_phv_ade.prompt")
        zh, fl = m.zh_system, list(m.function_list)
    elif task == "ccbi":
        m = importlib.import_module("tasks.pharm_ccbi.prompt")
        zh, fl = m.zh_system, list(m.function_list)
        if with_summarize:
            fl.append(TopicSummarization(llm))
    elif task in ("dts_symptom", "dts_treatment", "dts_need"):
        m = importlib.import_module("tasks.pharm_dts.prompt")
        zh = m.PROMPTS[task]
        fl = list(m.FUNCTION_LIST)
    else:
        raise ValueError(
            f"Unknown task {task!r}. Use: dcv | phv_ade | ccbi | dts_symptom | dts_treatment | dts_need"
        )
    return zh, fl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="Meta-Llama-3.1-8B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://0.0.0.0:8007/v1",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api_key", type=str, default="mysecrettoken123", help="API key"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results",
        help="Directory for JSON results",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max queries (0 = no limit)"
    )
    parser.add_argument(
        "--with_summarize",
        action="store_true",
        help="For ccbi only: add topic_summarization tool (needs extra LLM calls)",
    )
    parser.add_argument(
        "--trace-log",
        type=str,
        default="",
        help="Enable JSONL trace: file path, or 1/auto for results/traces/<task>_<model>.jsonl. "
        "If empty, set env PHAR_BENCHMARK_TRACE to the same values to enable. "
        "Trace previews are not truncated by default; set PHAR_TRACE_MAX_CHARS (positive int) to cap.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(
        args.output_path, f"{args.task}_{args.model.split('/')[-1]}.json"
    )
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    llm = {
        "model_server": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
    }
    zh, function_list = load_prompt_and_tools(args.task, args.with_summarize, llm)

    trace_arg = (args.trace_log or os.environ.get("PHAR_BENCHMARK_TRACE", "") or "").strip()
    trace_writer = None
    if trace_arg:
        if trace_arg.lower() in ("1", "auto", "yes", "true"):
            trace_path = default_trace_path(args.output_path, args.task, args.model)
        else:
            trace_path = Path(trace_arg)
        trace_writer = BenchmarkTraceWriter(
            trace_path,
            task=args.task,
            model=args.model,
            eval_mode="agent",
        )

    bot = SocialMediaAgent(
        system_message=zh, function_list=function_list, llm=llm, trace_writer=trace_writer
    )

    queries = []
    with open(args.query_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    if args.limit > 0:
        queries = queries[: args.limit]

    try:
        for idx, row in enumerate(tqdm(queries)):
            qid = str(row.get("id", row.get("qid", idx)))
            query = row["query"]
            if qid in results:
                continue
            messages = []
            print(f"{Fore.RED + Style.BRIGHT}用户请求:{Style.RESET_ALL} {query}")
            messages.append({"role": "user", "content": query})
            response_plain_text = ""
            print(f"{Fore.CYAN + Style.BRIGHT}智能体回应:{Style.RESET_ALL}")
            if trace_writer:
                trace_writer.start_query(qid, query)
            _reset_data_folder_read_budget(bot)
            attempts = 0
            last_exc: BaseException | None = None
            for _retry in range(5):
                attempts += 1
                try:
                    for response in bot.run(messages=messages, lang="zh"):
                        response_plain_text = typewriter_print(
                            response, response_plain_text
                        )
                    messages.extend(response)
                    print("\n")
                    result = remove_think_tags(messages[-1]["content"])
                    results[qid] = result
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    last_exc = None
                    break
                except ModelServiceError as e:
                    last_exc = e
                    if trace_writer:
                        trace_writer.log_retry(_retry, e)
                    _reset_data_folder_read_budget(bot)
                    sleep(60)
                    continue
                except Exception as e:
                    last_exc = e
                    if trace_writer:
                        trace_writer.log_retry(_retry, e)
                    _reset_data_folder_full(bot)
                    messages = []
                    messages.append({"role": "user", "content": query})
                    response_plain_text = ""
                    print(f"{Fore.CYAN + Style.BRIGHT}智能体回应:{Style.RESET_ALL}")
                    continue
            if qid not in results:
                results[qid] = ""
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            if trace_writer:
                final = str(results.get(qid, ""))
                if final.strip():
                    st = "ok"
                elif last_exc is not None:
                    st = "error"
                else:
                    st = "empty"
                extra = {}
                if last_exc is not None:
                    extra["last_error"] = _trunc_err(last_exc)
                trace_writer.end_query(
                    status=st,
                    final_text=final,
                    retries_used=max(0, attempts - 1),
                    extra=extra or None,
                )

            _reset_data_folder_full(bot)
    finally:
        if trace_writer:
            trace_writer.close()


if __name__ == "__main__":
    main()
