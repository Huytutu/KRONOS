# VinDr-CXR VQA Evaluation — Build Plan

Spec: `eval_vindr_vqa_SPEC.md`

## Tasks

### Task 1: Gemini API client
**File:** `src/llm/gemini_client.py`
**What:** Thin wrapper around `google-generativeai` SDK. `complete(prompt, model)` function, API key from `GEMINI_API_KEY` env var, exponential backoff on rate limit (max 3 retries).
**Depends on:** nothing
**Acceptance:** Unit test with mocked `genai` calls passes. Function returns text string.

### Task 2: VinDr VQA grading metrics
**File:** `src/eval/vindr_vqa_metrics.py`
**What:** `judge_answer(question, prediction, ground_truth, llm_fn) -> int` and `grade_batch(items, predictions, llm_fn) -> dict` with overall/by_type/by_difficulty breakdown.
**Depends on:** Task 1 (uses llm_fn signature)
**Acceptance:** Tests verify judge parsing (CORRECT→1, INCORRECT→0, robustness) and aggregation math.

### Task 3: CLI eval runner
**File:** `scripts/eval_vindr_vqa.py`
**What:** CLI entry point following `eval_multihop.py` pattern. Loads data, inits KRONOS pipeline, runs predictions, calls Gemini judge, writes JSON report + prints table.
**Depends on:** Task 1, Task 2
**Acceptance:** Smoke test with mocked pipeline + mocked Gemini writes valid JSON report.

### Task 4: Tests
**File:** `tests/test_eval_vindr_vqa.py`
**What:** All unit tests: judge_answer correct/incorrect/robustness, grade_batch aggregation, CLI smoke test. All `not gpu` marked.
**Depends on:** Task 1, Task 2, Task 3 (tests written alongside each task, collected here)
**Acceptance:** `pytest tests/test_eval_vindr_vqa.py -m "not gpu"` passes.
