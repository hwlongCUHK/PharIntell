#!/usr/bin/env python3
"""DCV: extract structured label from run_phar_task JSON (SoMe MID_extraction style)."""
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

VALID_LABELS = frozenset({"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"})


def extract_label_rule(raw: str) -> str | None:
    obj = first_json_object(raw)
    if not obj:
        return None
    lab = obj.get("label")
    if isinstance(lab, str) and lab.strip().upper() in VALID_LABELS:
        return lab.strip().upper()
    if isinstance(lab, str):
        u = lab.strip().upper().replace(" ", "_")
        if u in VALID_LABELS:
            return u
    return None


def extract_label_llm(client, model: str, raw: str) -> str:
    prompt = """Read the assistant text. Output exactly one token: SUPPORTED, REFUTED, NOT_ENOUGH_INFO, or ERROR.
The text should contain a JSON verdict for a drug claim; infer the label if needed. No quotes, no explanation."""
    user = f"Text:\n{raw[:12000]}"
    for _ in range(8):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        ans = strip_think_tags((resp.choices[0].message.content or "").strip()).upper()
        for tok in VALID_LABELS | {"ERROR"}:
            if tok in ans.split()[:3] or ans == tok:
                if tok == "ERROR":
                    return "ERROR"
                return tok
        if ans in VALID_LABELS:
            return ans
    return "ERROR"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result_path",
        type=str,
        default="results/phar_tasks",
        help="Directory containing dcv_<model>.json from run_phar_task.py",
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default="scores/phar_dcv",
        help="Directory to write extracted labels per model file",
    )
    ap.add_argument(
        "--setting_path",
        type=str,
        default="eval_scripts/settings.json",
    )
    ap.add_argument(
        "--use_llm_fallback",
        action="store_true",
        help="Call eval LLM when JSON parse fails",
    )
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

    for json_path in tqdm(list_result_jsons(result_dir, "dcv")):
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
        out_file = out_dir / json_path.name
        results: dict[str, str] = {}
        if out_file.exists():
            results = json.loads(out_file.read_text(encoding="utf-8"))

        for qid, samples in tqdm(raw_data.items(), leave=False):
            if qid in results:
                continue
            if not isinstance(samples, str):
                samples = json.dumps(samples, ensure_ascii=False)

            lab = extract_label_rule(samples)
            if lab is None and args.use_llm_fallback and client:
                lab = extract_label_llm(client, settings["model"], samples)
            elif lab is None:
                lab = "ERROR"

            results[qid] = lab

        out_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
