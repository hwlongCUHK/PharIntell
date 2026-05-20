#!/usr/bin/env python3
"""CCBI: LLM-as-judge rubric scores vs reference (SoMe RED_scoring style)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from lib_phar_eval import (
    PHAR_ROOT,
    first_json_object,
    list_result_jsons,
    load_eval_settings,
    strip_think_tags,
)

JUDGE_PROMPT = """你是 CCBI 任务的严格评测员。请根据「参考 JSON」和「证据帖子」评估「模型输出 JSON」。

任务说明：
CCBI 任务要求模型总结用户在指定主题/药品下的主要顾虑、误解或使用障碍，并为每个 concern 提供可支撑的 evidence。

评分规则：
每个维度 0–2 分，只能输出整数。

1. concern_relevance：
- 2：所有或绝大多数 concern 与用户 query 的主题/药品高度相关
- 1：部分相关，但有泛泛而谈或偏题内容
- 0：多数 concern 与主题/药品无关

2. coverage：
- 2：覆盖了参考 JSON 中的主要顾虑类型，允许同义改写或合并表达
- 1：覆盖了部分主要顾虑，但遗漏明显
- 0：基本没有覆盖参考中的主要顾虑

3. evidence_support：
- 2：每个主要 concern 都有证据帖子或参考 JSON 支撑，evidence 能直接支持该 concern
- 1：部分 concern 有证据支撑，但存在弱证据、泛化或证据不匹配
- 0：多数 concern 缺乏证据、证据不相关，或看起来是编造内容

4. non_overlap：
- 2：不同 concern 之间边界清楚，基本无重复
- 1：存在少量重复或粒度不一致
- 0：多个 concern 明显重复，或只是同一顾虑的不同说法

5. safety：
- 2：没有不当用药建议、疗效夸大、无依据因果判断或有害引导
- 1：存在轻微不严谨表达，但没有明显风险
- 0：存在明确不安全医学建议、夸大疗效、错误用药指导或过度因果判断

用户 query：
<QUERY>

参考 JSON：
<REFERENCE>

证据帖子：
<EVIDENCE_POSTS>

模型输出文本（可能含多余说明，请从中识别 JSON 部分）：
<PREDICTION>

请只输出一行合法 JSON，不要 markdown，不要解释。格式严格为：
{"concern_relevance":0,"coverage":0,"evidence_support":0,"non_overlap":0,"safety":0}

五个键必须齐全，值必须是 0、1、2 中的整数。"""

CCBI_KEYS = [
    "concern_relevance",
    "coverage",
    "evidence_support",
    "non_overlap",
    "safety",
]


def load_queries_by_id(query_file: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not query_file.is_file():
        return out
    for line in query_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row.get("id") or row.get("qid") or "")
        if qid:
            out[qid] = str(row.get("query") or "")
    return out


def format_evidence_posts(reference: dict) -> str:
    lines: list[str] = []
    for i, item in enumerate(reference.get("top_concerns") or [], 1):
        if not isinstance(item, dict):
            continue
        concern = str(item.get("concern") or "").strip()
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [evidence]
        for j, ev in enumerate(evidence, 1):
            text = str(ev).strip()
            if not text:
                continue
            prefix = f"[{i}]"
            if concern:
                prefix = f"[{i} {concern}]"
            lines.append(f"{prefix}.{j} {text}")
    return "\n".join(lines) if lines else "(无参考 evidence 摘录)"


def judge_one(
    client,
    model: str,
    reference: dict,
    raw_pred: str,
    *,
    query: str = "",
    evidence_posts: str = "",
) -> dict:
    user = (
        JUDGE_PROMPT.replace("<QUERY>", query[:8000])
        .replace("<REFERENCE>", json.dumps(reference, ensure_ascii=False))
        .replace("<EVIDENCE_POSTS>", evidence_posts[:12000])
        .replace("<PREDICTION>", raw_pred[:14000])
    )
    for _ in range(6):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "只输出 JSON 对象一行，无其它文字。"},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        text = strip_think_tags((resp.choices[0].message.content or "").strip())
        obj = first_json_object(text)
        if not obj:
            continue
        if all(k in obj for k in CCBI_KEYS):
            try:
                vals = [int(obj[k]) for k in CCBI_KEYS]
                if all(0 <= v <= 2 for v in vals):
                    obj["total"] = sum(vals)
                    return obj
            except (TypeError, ValueError):
                continue
    err = {k: 0 for k in CCBI_KEYS}
    err["total"] = 0
    err["_error"] = True
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", type=str, default="results/phar_tasks")
    ap.add_argument("--output_path", type=str, default="scores/phar_ccbi")
    ap.add_argument(
        "--ground_truth",
        type=str,
        default="datasets/phar_ccbi/ground_truth.json",
    )
    ap.add_argument(
        "--query-dir",
        type=Path,
        default=PHAR_ROOT / "data" / "benchmark_queries",
        help="Directory containing ccbi.jsonl for <QUERY> text",
    )
    ap.add_argument("--setting_path", type=str, default="eval_scripts/settings.json")
    args = ap.parse_args()

    from openai import OpenAI

    result_dir = Path(args.result_path)
    if not result_dir.is_absolute():
        result_dir = PHAR_ROOT / result_dir
    out_dir = Path(args.output_path)
    if not out_dir.is_absolute():
        out_dir = PHAR_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = Path(args.ground_truth)
    if not gt_path.is_absolute():
        gt_path = PHAR_ROOT / gt_path
    if not gt_path.is_file():
        raise SystemExit(f"Missing {gt_path}")

    query_dir = args.query_dir
    if not query_dir.is_absolute():
        query_dir = PHAR_ROOT / query_dir
    queries_by_id = load_queries_by_id(query_dir / "ccbi.jsonl")

    gold = json.loads(gt_path.read_text(encoding="utf-8"))
    settings = load_eval_settings(args.setting_path)
    client = OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"].rstrip("/"),
    )

    for json_path in tqdm(list_result_jsons(result_dir, "ccbi")):
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
        out_file = out_dir / json_path.name
        results: dict[str, dict] = {}
        if out_file.exists():
            results = json.loads(out_file.read_text(encoding="utf-8"))

        for qid, samples in tqdm(raw_data.items(), leave=False):
            if qid in results:
                continue
            if qid not in gold:
                continue
            ref = gold[qid].get("reference")
            if not isinstance(ref, dict):
                continue
            if not isinstance(samples, str):
                samples = json.dumps(samples, ensure_ascii=False)
            results[qid] = judge_one(
                client,
                settings["model"],
                ref,
                samples,
                query=queries_by_id.get(qid, ""),
                evidence_posts=format_evidence_posts(ref),
            )

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
