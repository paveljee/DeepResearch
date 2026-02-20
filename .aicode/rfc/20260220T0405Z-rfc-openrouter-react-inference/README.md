# RFC: OpenRouter ReAct Inference Pathway for DeepResearch

## 1) Context and Objective

This RFC documents a minimal-change implementation of an OpenRouter-first ReAct inference path, aligned with the main README workflow and preserving existing behavior where possible.

Requested deliverables:

1. Create `inference/run_react_infer_openrouter.sh` to run ReAct inference through OpenRouter (no local vLLM serving).
2. Keep changes/deviations from `inference/run_react_infer.sh` to the strict minimum needed.
3. Audit dependencies required for this OpenRouter pathway and create `requirements_openrouter.txt` that excludes local inference stack (e.g., `vllm`, CUDA, torch serving path).
4. Record all findings and decisions here.

---

## 2) Baseline Workflow Review (Current Code)

### 2.1 Main script in README

The README’s main flow points to `inference/run_react_infer.sh`, which:

- loads `.env`,
- validates `MODEL_PATH`,
- launches 8 local `vllm serve` processes on ports 6001-6008,
- waits for readiness via `/v1/models`,
- runs `python -u run_multi_react.py ...`.

### 2.2 Runtime control flow

`run_multi_react.py`:

- loads dataset (`.json` or `.jsonl`),
- creates per-rollout task list,
- assigns each task a `planning_port` (round-robin over 6001-6008),
- invokes `MultiTurnReactAgent._run(...)` in a thread pool,
- writes per-rollout jsonl outputs.

`react_agent.py`:

- orchestrates ReAct loop,
- uses `call_server(...)` for LLM calls,
- originally points OpenAI client to local `http://127.0.0.1:{planning_port}/v1` with key `EMPTY`,
- includes commented OpenRouter note for reasoning concatenation,
- uses local `AutoTokenizer.from_pretrained(self.llm_local_path)` for token counting.

### 2.3 Gap for OpenRouter

README mentions OpenRouter usage, but operationally this path was not fully runnable out-of-the-box because:

- no dedicated launcher script for OpenRouter mode,
- `call_server` defaulted to local vLLM endpoint only,
- token counting depended on local model path tokenizer load, which may be unnecessary/problematic for API-only setup.

---

## 3) Implemented Changes

## 3.1 New script: `inference/run_react_infer_openrouter.sh`

Design goal: mirror the structure of `run_react_infer.sh` while removing only local-serving steps.

What it does:

- loads `.env` exactly like original script,
- validates `OPENROUTER_API_KEY`,
- defines defaults:
  - `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
  - `OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b`
- skips vLLM startup and port readiness checks,
- runs the same `run_multi_react.py` command shape with existing env-driven knobs (`DATASET`, `OUTPUT_PATH`, `MAX_WORKERS`, `TEMPERATURE`, `PRESENCE_PENALTY`, `WORLD_SIZE`, `RANK`, `ROLLOUT_COUNT`),
- passes `USE_OPENROUTER=true` and OpenRouter credentials as environment variables.

Why this is minimal:

- no structural change to `run_multi_react.py`,
- keeps existing rollout/parallelization behavior intact,
- only removes components specific to local vLLM serving.

## 3.2 Minimal `react_agent.py` adaptation

Only necessary runtime branching was added:

1. **Endpoint/key selection in `call_server`**
   - when `USE_OPENROUTER=true`:
     - `api_key = OPENROUTER_API_KEY`
     - `base_url = OPENROUTER_BASE_URL` (default supported in launcher)
   - otherwise keep original local-vLLM path behavior.

2. **Reasoning-content concatenation for OpenRouter**
   - implemented the previously commented behavior safely:
     - if response message has non-empty `.reasoning`, prepend as:
       `<think>...</think>` + assistant `content`.

3. **Token counting fallback for API-only mode**
   - when `USE_OPENROUTER=true`, use `tiktoken` (`cl100k_base`) over serialized messages,
   - otherwise retain original local `transformers` tokenizer path.

Rationale:

- preserves original local inference behavior,
- avoids requiring local model tokenizer artifacts for API-only mode,
- keeps rest of agent/tool loop untouched.

---

## 4) OpenRouter Dependency Audit

Scope audited:

- `inference/run_multi_react.py`
- `inference/react_agent.py`
- tool chain imported by `react_agent.py`:
  - `tool_search.py`
  - `tool_visit.py`
  - `tool_scholar.py`
  - `tool_python.py`
  - `tool_file.py`
  - `inference/file_tools/*` transitively imported by file tool

### 4.1 What is intentionally excluded

Any local serving/runtime stack for model hosting:

- `vllm`
- CUDA/NVIDIA runtime packages
- `torch` serving path dependencies tied to local model inference
- related high-cost local inference ecosystem packages

### 4.2 What is included

`requirements_openrouter.txt` includes only packages needed for:

- ReAct runner + API LLM calls (`qwen-agent`, `openai`, `requests`, `json5`, `tqdm`, `python-dotenv`),
- token counting used in this pathway (`tiktoken`, `transformers`),
- enabled tools (search/visit/python/file) and file parsing helpers,
- sandbox execution tool (`sandbox-fusion`),
- optional file/video helper imports used by parser path.

Note:
- This file is pathway-focused and does not include local vLLM model-hosting dependencies.
- It retains tool-path dependencies because tool modules are imported by the ReAct agent in this code path.

---

## 5) Usage Notes for OpenRouter Path

1. Install dependencies:

```bash
pip install -r requirements_openrouter.txt
```

2. Configure `.env`:

- required: `OPENROUTER_API_KEY`
- recommended defaults (optional override):
  - `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
  - `OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b`
- existing runner vars still apply: `DATASET`, `OUTPUT_PATH`, `MAX_WORKERS`, etc.

3. Run:

```bash
bash inference/run_react_infer_openrouter.sh
```

---

## 6) Validation Performed

- Shell syntax validation for new launcher.
- Python syntax validation for modified `react_agent.py`.
- Git diff inspection to confirm minimal-touch intent.

(Functional live API run was not executed in this RFC because no OpenRouter key was provided in this environment.)

---

## 7) Deviations vs Original Script (Explicit)

Compared to `inference/run_react_infer.sh`, the new script differs only where needed:

1. Removes local vLLM startup (8 GPU-backed servers) and readiness waits.
2. Validates OpenRouter API key instead of model path.
3. Injects OpenRouter mode/env vars for runtime branch in `react_agent.py`.
4. Uses OpenRouter model id as `--model` argument (for output naming + API model selection).

All other control knobs and `run_multi_react.py` invocation shape remain consistent.

