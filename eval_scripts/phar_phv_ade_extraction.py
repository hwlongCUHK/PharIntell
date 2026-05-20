#!/usr/bin/env python3
"""PhV-ADE: extract ade_class (+ optional fields) from run_phar_task output."""
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

ADE_CLASSES = frozenset({"ADE-positive", "ADE-negative", "Unclear"})


def has_ade_to_class(has_ade: bool | None, explicit: str | None) -> str | None:
    if explicit and explicit in ADE_CLASSES:
        return explicit
    if has_ade is True:
        return "ADE-positive"
    if has_ade is False:
        return "ADE-negative"
    if has_ade is None:
        return "Unclear"
    return None


def extract_rule(raw: str) -> dict | None:
    obj = first_json_object(raw)
    if not obj:
        return None
    explicit = obj.get("ade_class")
    if isinstance(explicit, str) and explicit in ADE_CLASSES:
        return {"ade_class": explicit, "raw": obj}
    ha = obj.get("has_ADE")
    if isinstance(ha, bool):
        cls = has_ade_to_class(ha, None)
        if cls:
            return {
                "ade_class": cls,
                "has_ADE": ha,
                "adverse_events": obj.get("adverse_events") or [],
                "evidence": obj.get("evidence", ""),
            }
    return None


def extract_llm(client, model: str, raw: str) -> dict:
    from openai import OpenAI

    prompt = """Output exactly one line, one of: ADE-positive, ADE-negative, Unclear
Based on whether the user post reports adverse drug events related to the drug."""
    user = raw[:12000]
    for _ in range(8):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        ans = strip_think_tags((resp.choices[0].message.content or "").strip())
        if "ADE-positive" in ans:
            return {"ade_class": "ADE-positive", "adverse_events": [], "evidence": ""}
        if "ADE-negative" in ans:
            return {"ade_class": "ADE-negative", "adverse_events": [], "evidence": ""}
        if "Unclear" in ans:
            return {"ade_class": "Unclear", "adverse_events": [], "evidence": ""}
    return {"ade_class": "Unclear", "adverse_events": [], "evidence": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", type=str, default="results/phar_tasks")
    ap.add_argument("--output_path", type=str, default="scores/phar_phv_ade")
    ap.add_argument("--setting_path", type=str, default="eval_scripts/settings.json")
    ap.add_argument("--use_llm_fallback", action="store_true")
    args = ap.parse_args()

    result_dir = Path(args.result_path)
    if not result_dir.is_absolute():
        result_dir = PHAR_ROOT / result_dir
    out_dir = Path(args.output_path)
    if not out_dir.is_absolute():
        out_dir = PHAR_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_eval_settings(args.setting_path)
    client = None
    if args.use_llm_fallback:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"].rstrip("/"),
        )

    for json_path in tqdm(list_result_jsons(result_dir, "phv_ade")):
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
        out_file = out_dir / json_path.name
        results: dict[str, dict] = {}
        if out_file.exists():
            results = json.loads(out_file.read_text(encoding="utf-8"))

        for qid, samples in tqdm(raw_data.items(), leave=False):
            if qid in results:
                continue
            if not isinstance(samples, str):
                samples = json.dumps(samples, ensure_ascii=False)
            row = extract_rule(samples)
            if row is None and args.use_llm_fallback and client:
                row = extract_llm(client, settings["model"], samples)
            elif row is None:
                row = {"ade_class": "Unclear", "adverse_events": [], "evidence": ""}
            results[qid] = row

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
