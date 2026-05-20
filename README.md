# PharIntell

Pharmaceutical social-media intelligence agents for **PharmXHS** benchmark tasks: **DCV**, **PhV-ADE**, **CCBI**, and **DTS** (symptom / treatment / need). This repo ships the agent runtime, tools, evaluation scripts, and benchmark gold labels.

## Layout

- `qwen_agent/` — vendored [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) framework (`SocialMediaAgent`, `BaseTool`, `BasicAgent`).
- `agent.py`, `config.py`, `run_phar_task.py` — agent entry and configuration.
- `tools/` — `post_search`, `data_folder`, `topic_clustering`, `topic_summarization`, `post_retrieve`, `knowledge_retrieve`.
- `tasks/pharm_*` — per-task system prompts and tool lists.
- `database/` — knowledge base JSON, sample queries, smoke posts; `emb_data/` and `raw_data/` for generated assets.
- `datasets/phar_*/ground_truth.json` — benchmark gold (`query_id` must match query JSONL `id`).
- `eval_scripts/` — extraction and LLM-as-judge scoring.
- `scripts/` — `csv_to_phar_jsonl.py`, `build_knowledge_emb.py`, `build_post_emb.py`.

## Install

```bash
cd PharIntell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables:

| Variable | Purpose |
|----------|---------|
| `PHAR_EMBEDDING_MODEL` | Embedding model path or Hub id (default: `Qwen/Qwen3-Embedding-0.6B`) |
| `PHAR_EMBEDDING_DEVICE` | `cuda` or `cpu` (auto-detect GPU if unset) |
| `PHAR_KNOWLEDGE_JSON` / `PHAR_KNOWLEDGE_NPY` | Override knowledge base paths |
| `PHAR_MAX_LLM_CALL_PER_RUN` | Max LLM rounds per query (default `20`) |
| `PHAR_SKIP_AUTO_POST_EMB` | Skip automatic `.npy` creation in `post_search` |
| `PHAR_EVAL_MODEL` / `PHAR_EVAL_BASE_URL` / `PHAR_EVAL_API_KEY` | Override eval judge API |

## Import check

```bash
cd PharIntell
PYTHONPATH=. python -c "from agent import SocialMediaAgent; import tools; import tools.en; print('ok')"
```

## Prepare knowledge embeddings (DCV / knowledge-augmented tasks)

```bash
PYTHONPATH=. python scripts/build_knowledge_emb.py
```

Requires `database/knowledge_data/knowledge_base.json`. Writes `database/emb_data/knowledge_base.npy`.

## Prepare post data

Convert your CSV export to JSONL (internal field keys for `post_search`):

```bash
PYTHONPATH=. python scripts/csv_to_phar_jsonl.py \
  --csv /path/to/drug_classified.csv \
  --corpus brand \
  --out-dir database/raw_data
```

For CCBI/DTS, precompute post-window embeddings (folder name = `{corpus}_{start}_{end}` with spaces/colons escaped on disk):

```bash
PYTHONPATH=. python scripts/build_post_emb.py \
  --jsonl database/raw_data/smoke_posts.jsonl \
  --folder-name "smoke_2025-03-01 00:00:00_2025-04-30 23:59:59"
```

`post_search` writes folder keys as `{corpus}_{start_time}_{end_time}`; do not insert extra underscores in timestamps so `topic_clustering` can split the name into three segments.

## Run a task

You need an **OpenAI-compatible** inference API (`--base_url`, `--api_key`, `--model`).

```bash
PYTHONPATH=. python run_phar_task.py \
  --task dcv \
  --query_file database/sample_queries/dcv.jsonl \
  --limit 1 \
  --base_url http://127.0.0.1:8007/v1 \
  --api_key your-api-key \
  --model your-model-name
```

Supported tasks: `dcv`, `phv_ade`, `ccbi`, `dts_symptom`, `dts_treatment`, `dts_need`.  
CCBI: add `--with_summarize` to register `TopicSummarization` (extra LLM calls).

Results are written to `results/<task>_<model>.json` by default.

### Task tools

| Task | Default tools |
|------|----------------|
| DCV | `knowledge_retrieve` |
| PhV-ADE | none (LLM only) |
| CCBI | `post_search` → `topic_clustering` → `data_folder` (optional summarization) |
| DTS* | `post_search` → `post_retrieve` → `topic_clustering` → `data_folder` |

## Evaluation

1. Copy eval API settings:

   ```bash
   cp eval_scripts/settings.json.example eval_scripts/settings.json
   # edit model / base_url / api_key, or use PHAR_EVAL_* env vars
   ```

2. Run inference (above), then task-specific scripts from `PharIntell/`:

   **DCV / PhV-ADE** (rule extraction + optional LLM fallback):

   ```bash
   python eval_scripts/phar_dcv_extraction.py [--use_llm_fallback]
   python eval_scripts/phar_dcv_compute_score.py

   python eval_scripts/phar_phv_ade_extraction.py [--use_llm_fallback]
   python eval_scripts/phar_phv_ade_compute_score.py
   ```

   **CCBI / DTS** (LLM-as-judge):

   ```bash
   python eval_scripts/phar_ccbi_scoring.py
   python eval_scripts/phar_ccbi_compute_score.py

   python eval_scripts/phar_dts_scoring.py --variant symptom
   python eval_scripts/phar_dts_compute_score.py --variant symptom
   ```

   Aggregate paper-style table (if you have scores for multiple tasks):

   ```bash
   python eval_scripts/aggregate_paper_results.py --model your-model-name
   ```

Gold labels live under `datasets/phar_<task>/ground_truth.json`.

## License

MIT License with third-party notice for Qwen-Agent (see [LICENSE](LICENSE)).
