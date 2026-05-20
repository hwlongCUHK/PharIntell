#!/usr/bin/env python3
"""DTS subtasks: LLM-as-judge (symptom / treatment / need)."""
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

PROMPTS = {
    "symptom": """你是 DTS-Symptom 任务的严格评测员。请根据「参考 JSON」评估「模型输出」。

任务目标：
DTS-Symptom 要求模型总结指定主题下的主要症状表现或症状共现模式，并提供能支撑每个症状主题的 evidence。不要把治疗方法、用药行为或未满足需求当作症状主题。

评分规则：每个维度 0–2 分，只能输出整数。

1. symptom_relevance：
- 2：输出的症状主题均与 query/参考主题高度相关
- 1：部分相关，但混入治疗方法、需求或泛泛内容
- 0：多数内容不是症状主题或明显跑题

2. coverage：
- 2：覆盖参考 JSON 中的主要症状类型，允许同义改写或合理合并
- 1：覆盖部分主要症状，但遗漏明显
- 0：基本没有覆盖参考中的主要症状

3. evidence_support：
- 2：主要症状主题均有参考 JSON 或 evidence 支撑，证据能直接支持对应主题
- 1：部分主题有证据支撑，但存在弱证据、泛化或证据不匹配
- 0：多数主题缺乏证据、证据不相关，或疑似编造

4. non_overlap：
- 2：不同症状主题边界清楚，基本无重复
- 1：存在少量重复或粒度不一致
- 0：多个主题明显重复，或只是同一症状的不同说法

5. clarity：
- 2：输出结构清晰、主题命名具体、易于理解
- 1：基本可读，但结构或命名不够清楚
- 0：结构混乱，难以判断具体主题

参考 JSON：
<REFERENCE>

模型输出文本（可能含多余说明，请从中识别 JSON 部分）：
<PREDICTION>

只输出一行合法 JSON，不要 markdown，不要解释。格式严格为：
{"symptom_relevance":0,"coverage":0,"evidence_support":0,"non_overlap":0,"clarity":0}
五个键必须齐全，值必须是 0、1、2 中的整数。""",
    "treatment": """你是 DTS-Treatment 任务的严格评测员。请根据「参考 JSON」评估「模型输出」。

任务目标：
DTS-Treatment 要求模型总结指定主题下用户提到的自我管理方式、治疗尝试、用药行为或护理方法，并提供能支撑每个治疗主题的 evidence。不要把单纯症状表现或未满足需求当作治疗主题。

评分规则：每个维度 0–2 分，只能输出整数。

1. treatment_relevance：
- 2：输出的治疗/处理方式均与 query/参考主题高度相关
- 1：部分相关，但混入症状、需求或泛泛内容
- 0：多数内容不是治疗或处理方式，或明显跑题

2. coverage：
- 2：覆盖参考 JSON 中的主要治疗/处理方式，允许同义改写或合理合并
- 1：覆盖部分主要方式，但遗漏明显
- 0：基本没有覆盖参考中的主要方式

3. evidence_support：
- 2：主要治疗主题均有参考 JSON 或 evidence 支撑，证据能直接支持对应主题
- 1：部分主题有证据支撑，但存在弱证据、泛化或证据不匹配
- 0：多数主题缺乏证据、证据不相关，或疑似编造

4. non_overlap：
- 2：不同治疗主题边界清楚，基本无重复
- 1：存在少量重复或粒度不一致
- 0：多个主题明显重复，或只是同一方式的不同说法

5. safety：
- 2：没有不当用药建议、疗效夸大、无依据因果判断或有害引导
- 1：存在轻微不严谨表达，但没有明显风险
- 0：存在明确不安全用药建议、夸大疗效、错误治疗指导或过度因果判断

参考 JSON：
<REFERENCE>

模型输出文本（可能含多余说明，请从中识别 JSON 部分）：
<PREDICTION>

只输出一行合法 JSON，不要 markdown，不要解释。格式严格为：
{"treatment_relevance":0,"coverage":0,"evidence_support":0,"non_overlap":0,"safety":0}
五个键必须齐全，值必须是 0、1、2 中的整数。""",
    "need": """你是 DTS-Need 任务的严格评测员。请根据「参考 JSON」评估「模型输出」。

任务目标：
DTS-Need 要求模型总结指定主题下用户表达的未满足需求、困惑、决策障碍或希望获得的帮助，并提供能支撑每个需求主题的 evidence。不要把单纯症状或治疗方法当作未满足需求。

评分规则：每个维度 0–2 分，只能输出整数。

1. need_relevance：
- 2：输出的需求主题均与 query/参考主题高度相关
- 1：部分相关，但混入症状、治疗方式或泛泛内容
- 0：多数内容不是未满足需求，或明显跑题

2. coverage：
- 2：覆盖参考 JSON 中的主要需求类型，允许同义改写或合理合并
- 1：覆盖部分主要需求，但遗漏明显
- 0：基本没有覆盖参考中的主要需求

3. evidence_support：
- 2：主要需求主题均有参考 JSON 或 evidence 支撑，证据能直接支持对应主题
- 1：部分主题有证据支撑，但存在弱证据、泛化或证据不匹配
- 0：多数主题缺乏证据、证据不相关，或疑似编造

4. non_overlap：
- 2：不同需求主题边界清楚，基本无重复
- 1：存在少量重复或粒度不一致
- 0：多个主题明显重复，或只是同一需求的不同说法

5. clarity：
- 2：输出结构清晰、需求命名具体、易于理解
- 1：基本可读，但结构或命名不够清楚
- 0：结构混乱，难以判断具体需求

参考 JSON：
<REFERENCE>

模型输出文本（可能含多余说明，请从中识别 JSON 部分）：
<PREDICTION>

只输出一行合法 JSON，不要 markdown，不要解释。格式严格为：
{"need_relevance":0,"coverage":0,"evidence_support":0,"non_overlap":0,"clarity":0}
五个键必须齐全，值必须是 0、1、2 中的整数。""",
}

KEY_SETS = {
    "symptom": ["symptom_relevance", "coverage", "evidence_support", "non_overlap", "clarity"],
    "treatment": ["treatment_relevance", "coverage", "evidence_support", "non_overlap", "safety"],
    "need": ["need_relevance", "coverage", "evidence_support", "non_overlap", "clarity"],
}


def judge_one(client, model: str, template: str, keys: list[str], reference: dict, raw_pred: str) -> dict:
    user = (
        template.replace("<REFERENCE>", json.dumps(reference, ensure_ascii=False)).replace(
            "<PREDICTION>", raw_pred[:14000]
        )
    )
    for _ in range(6):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "只输出 JSON 一行，无 markdown。"},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        text = strip_think_tags((resp.choices[0].message.content or "").strip())
        obj = first_json_object(text)
        if not obj:
            continue
        if all(k in obj for k in keys):
            try:
                vals = [int(obj[k]) for k in keys]
                if all(0 <= v <= 2 for v in vals):
                    obj["total"] = sum(vals)
                    return obj
            except (TypeError, ValueError):
                continue
    err = {k: 0 for k in keys}
    err["total"] = 0
    err["_error"] = True
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["symptom", "treatment", "need"],
        required=True,
    )
    ap.add_argument("--result_path", type=str, default="results/phar_tasks")
    ap.add_argument("--setting_path", type=str, default="eval_scripts/settings.json")
    args = ap.parse_args()

    from openai import OpenAI

    prefix = f"dts_{args.variant}"
    gt_name = f"datasets/phar_dts_{args.variant}/ground_truth.json"
    out_sub = f"scores/phar_dts_{args.variant}"

    result_dir = Path(args.result_path)
    if not result_dir.is_absolute():
        result_dir = PHAR_ROOT / result_dir
    out_dir = PHAR_ROOT / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = PHAR_ROOT / gt_name
    if not gt_path.is_file():
        raise SystemExit(f"Missing {gt_path}")

    gold = json.loads(gt_path.read_text(encoding="utf-8"))
    settings = load_eval_settings(args.setting_path)
    client = OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"].rstrip("/"),
    )
    keys = KEY_SETS[args.variant]
    template = PROMPTS[args.variant]

    for json_path in tqdm(list_result_jsons(result_dir, prefix)):
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
                client, settings["model"], template, keys, ref, samples
            )

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
