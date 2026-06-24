# VinDr-CXR VQA Evaluation — SPEC

Evaluate KRONOS full pipeline on the VinDr-CXR VQA dataset (17,597 questions,
6 types, 2 difficulty levels). Uses Google Gemini as LLM-as-judge to score
free-text answers against ground truth.

---

## 1. Objective

Run the full KRONOS pipeline (YOLO detect → parse question → tree search →
verifier) on VinDr-CXR VQA questions, then grade each predicted answer against
the ground truth using a Gemini LLM judge. Report accuracy overall and broken
down by question type and difficulty.

**Non-goals:**
- Training or fine-tuning any model.
- Baseline comparisons (VLM-only, CoT) — future work.
- BLEU/ROUGE/BERTScore — LLM judge is the primary metric.

---

## 2. Commands

```bash
conda activate medcxr

# Run eval (default: 50 samples)
python scripts/eval_vindr_vqa.py

# Run with limit
python scripts/eval_vindr_vqa.py --limit 100

# Full dataset
python scripts/eval_vindr_vqa.py --limit 0

# Custom paths
python scripts/eval_vindr_vqa.py --vqa data/vindr_cxr_vqa/vqa.json \
    --image-dir data/vindr_cxr_vqa/train --limit 50

# Run tests
pytest tests/ -k "vindr_vqa" -v
```

---

## 3. Project structure

```
scripts/eval_vindr_vqa.py         # CLI entry point (like eval_multihop.py)
src/eval/vindr_vqa_metrics.py     # grading logic: Gemini judge + aggregation
src/llm/gemini_client.py          # Google Gemini API client
tests/test_eval_vindr_vqa.py      # unit tests (mock Gemini, mock pipeline)
```

Existing files used read-only:
```
src/data/loaders.py               # load_vindr_vqa() already exists
src/pipeline.py                   # run() — full pipeline entry point
src/perception/detector.py        # YOLO detector
src/agent/medgemma.py             # MedGemma agent
src/ontology/dag.py               # OntologyDAG
```

---

## 4. Data flow

```
vqa.json ──load_vindr_vqa()──► List[QAItem]
                                   │
                              (--limit N)
                                   │
                    ┌──────────────┤
                    ▼              │
              QAItem.image         QAItem.question
                    │              │
                    ▼              ▼
              pipeline.run(image, question, dag, detector, agent)
                    │
                    ▼
              SearchResult.answer  (predicted)
                    │
                    ▼
              gemini_judge(prediction, ground_truth, question)
                    │
                    ▼
              score: 0 or 1  (correct / incorrect)
                    │
                    ▼
              aggregate by type, difficulty, overall
                    │
                    ▼
              results/vindr_vqa_report.json + stdout table
```

---

## 5. Components

### 5.1 `src/llm/gemini_client.py` — Gemini API client

Thin wrapper around `google-generativeai` SDK.

```python
def complete(prompt, model="gemini-2.0-flash"):
    """Send prompt to Gemini, return text response."""
```

- API key from `GEMINI_API_KEY` env var.
- No image support needed (judge only sees text).
- Retry on rate limit (simple exponential backoff, max 3 retries).

### 5.2 `src/eval/vindr_vqa_metrics.py` — Grading logic

**`judge_answer(question, prediction, ground_truth, llm_fn) -> int`**

Sends a structured prompt to Gemini asking it to judge whether the prediction
correctly answers the question, compared to the ground truth. Returns 1
(correct) or 0 (incorrect).

Judge prompt template:
```
You are a medical imaging expert grading a VQA system's answer.

Question: {question}
Ground truth answer: {ground_truth}
System prediction: {prediction}

Does the prediction correctly answer the question? Consider:
- For Yes/No questions: is the Yes/No part correct?
- For location questions: does it identify the correct region?
- For counting questions: is the count correct?
- For identification questions: does it name the correct finding(s)?

Reply with exactly "CORRECT" or "INCORRECT".
```

Parse response: `"CORRECT"` in response → 1, else → 0.

**`grade_batch(items, predictions, llm_fn) -> dict`**

Pairs each QAItem with its prediction, calls `judge_answer` for each, then
aggregates:

```python
{
    "n": int,
    "overall_accuracy": float,
    "by_type": {
        "Where": {"n": int, "accuracy": float},
        "Is_there": {"n": int, "accuracy": float},
        ...
    },
    "by_difficulty": {
        "Easy": {"n": int, "accuracy": float},
        "Medium": {"n": int, "accuracy": float},
    },
}
```

### 5.3 `scripts/eval_vindr_vqa.py` — CLI runner

Follows the same pattern as `scripts/eval_multihop.py`.

```
Args:
  --vqa         path to vqa.json (default: data/vindr_cxr_vqa/vqa.json)
  --image-dir   image directory (default: data/vindr_cxr_vqa/train)
  --limit       max questions to evaluate (default: 50, 0 = all)
  --weights     YOLO weights path (default: weights/yolov12s_vindr.pt)
  --model       MedGemma model path (default: weights/medgemma-4b-it)
  --out         output report path (default: results/vindr_vqa_report.json)
  --quantize    enable 4-bit quantization for MedGemma
```

Steps:
1. Load VQA data via `load_vindr_vqa()`, apply `--limit`.
2. Init detector (YOLO), agent (MedGemma), DAG.
3. For each QAItem: call `pipeline.run()` → get `SearchResult.answer`.
4. Collect all (item, prediction) pairs.
5. Call `grade_batch()` with Gemini judge.
6. Write JSON report + print summary table to stdout.

Stdout format:
```
VinDr-CXR VQA Evaluation (n=50)
  overall_accuracy       0.620

  By type:
    Where                0.550  (n=8)
    Is_there             0.700  (n=9)
    How_many             0.500  (n=7)
    Yes_No               0.750  (n=8)
    Which                0.600  (n=9)
    What                 0.580  (n=9)

  By difficulty:
    Easy                 0.700  (n=25)
    Medium               0.540  (n=25)

Report -> results/vindr_vqa_report.json
```

---

## 6. Testing strategy

### Unit tests (`tests/test_eval_vindr_vqa.py`)

1. **`test_judge_answer_correct`** — mock Gemini returning "CORRECT", verify
   `judge_answer` returns 1.

2. **`test_judge_answer_incorrect`** — mock Gemini returning "INCORRECT", verify
   returns 0.

3. **`test_judge_answer_parse_robustness`** — mock Gemini returning "CORRECT.
   The answer is right", verify still returns 1.

4. **`test_grade_batch_aggregation`** — 6 items (one per type), mock scores,
   verify overall/by_type/by_difficulty breakdown math.

5. **`test_cli_smoke`** — mock pipeline.run + Gemini, run CLI with `--limit 2`,
   verify JSON report written.

All tests: `pytest tests/ -k "vindr_vqa" -m "not gpu"` (no GPU, no real model).

---

## 7. Boundaries

### Always
- Use `load_vindr_vqa()` from `src/data/loaders.py` (already exists).
- Follow `eval_multihop.py` CLI pattern.
- Save predictions alongside scores in the report JSON (for debugging).
- Handle missing images gracefully (skip + warn).

### Ask first
- Adding new question types or metrics beyond the spec.
- Changing the judge prompt template.
- Adding batch/async Gemini calls for speed.

### Never
- Modify existing pipeline code (`pipeline.py`, `tree_search.py`, etc.).
- Hard-code API keys.
- Skip the LLM judge and use string matching for free-text answers.
