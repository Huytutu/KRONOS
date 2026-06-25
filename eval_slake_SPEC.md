# SLAKE VQA Evaluation — SPEC

Evaluate KRONOS on the SLAKE 1.0 X-Ray subset. Uses the SLAKE KG
(disease/organ knowledge base) as a separate lookup tool for KG-type
questions (cause, symptom, treatment, etc.). Exact-match grading.

---

## 1. Objective

Run the KRONOS pipeline on SLAKE's X-Ray questions and report accuracy.
For KG-type questions (1,174 in total, ~380 X-Ray), add a `slake_kg`
lookup tool so tree search can query disease attributes (cause, symptom,
treatment, prevention, location) and organ relations (belong_to, part_of,
function) from the SLAKE knowledge base.

**Scope:** X-Ray modality only (~2,808 questions out of 9,835).

**Data:**
- QA: `data/Slake1.0/test.json` (filter `modality == "X-Ray"`)
- KG: `data/Slake1.0/KG/en_disease.csv`, `en_organ.csv`, `en_organ_rel.csv`
- Images: `data/Slake1.0/imgs/<img_name>/source.jpg`

**SLAKE QA fields:**
```json
{
  "img_id": 1, "img_name": "xmlab1/source.jpg",
  "question": "What is the cause of the disease in this image?",
  "answer": "bacterial infection",
  "q_lang": "en", "location": "Lung", "modality": "X-Ray",
  "answer_type": "OPEN",  // or "CLOSED" (Yes/No)
  "content_type": "KG",   // Modality|Position|Organ|Abnormality|Size|KG|...
  "triple": ["vhead", "cause", "ktail"],
  "qid": 42
}
```

**SLAKE KG format** (`#`-separated CSV):
```
disease#attribute#value
Pneumonia#cause#bacterial, viral, or mycoplasma infection
Pneumonia#symptom#cough, fever, chest pain
Pneumonia#location#Lung
```

**Non-goals:**
- MRI/CT evaluation (YOLO detector is CXR-only).
- Modifying OntologyDAG or causal_kg.yaml.
- Fine-tuning any model.

---

## 2. Commands

```bash
conda activate medcxr

# Run eval (default: 50 samples)
python scripts/eval_slake.py --limit 50

# Full X-Ray subset
python scripts/eval_slake.py --limit 0

# Run tests
pytest tests/ -k "slake" -v -m "not gpu"
```

---

## 3. Project structure

New files:
```
src/knowledge/slake_kg.py         # SLAKE KG loader + lookup class
src/data/loaders.py               # add load_slake() function
scripts/eval_slake.py             # CLI runner
tests/test_eval_slake.py          # unit tests
```

Existing files (read-only):
```
src/pipeline.py                   # run() — full pipeline
src/perception/detector.py        # YOLO
src/agent/medgemma.py             # MedGemma
src/ontology/dag.py               # OntologyDAG (unchanged)
src/tools/dispatch.py             # tool dispatch (add slake_kg tool)
```

File to modify:
```
src/tools/dispatch.py             # register slake_kg tool
src/contracts.py                  # add "slake_kg" to ToolName literal
```

---

## 4. Data flow

```
test.json ──load_slake()──► List[QAItem]  (filter X-Ray, English)
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
                 │  (tree search may call slake_kg tool
                 │   for KG-type questions)
                 ▼
           SearchResult.answer
                 │
                 ▼
           exact_match(prediction, ground_truth)
                 │
                 ▼
           aggregate by content_type, answer_type, overall
                 │
                 ▼
           results/slake_report.json + stdout
```

---

## 5. Components

### 5.1 `src/knowledge/slake_kg.py` — SLAKE KG loader

```python
class SlakeKG:
    def __init__(self, kg_dir="data/Slake1.0/KG"):
        """Load en_disease.csv, en_organ.csv, en_organ_rel.csv."""

    def lookup(self, entity, relation):
        """Return value(s) for (entity, relation) pair.
        
        Examples:
          lookup("Pneumonia", "cause") → "bacterial, viral, or mycoplasma infection"
          lookup("Lung", "function") → "Breathe"
          lookup("Heart", "belong to") → "Circulatory System"
        Returns None if not found.
        """

    def diseases(self):
        """List all disease names."""

    def organs(self):
        """List all organ names."""
```

Internal storage: `dict[str, dict[str, str]]` — `{entity_lower: {relation: value}}`.

### 5.2 `src/data/loaders.py` — add `load_slake()`

```python
def load_slake(path, image_dir="data/Slake1.0/imgs", modality="X-Ray", lang="en"):
    """SLAKE 1.0: one QAItem per question, filtered by modality and language.
    
    meta keys: content_type, answer_type, triple, location, modality, img_id
    """
```

Add `"slake": load_slake` to the `LOADERS` dict.

### 5.3 `src/tools/dispatch.py` — register slake_kg tool

Add a `"slake_kg"` branch to the tool dispatcher. When called:

```python
# Action: slake_kg(entity="Pneumonia", relation="cause")
# → Observation(result="bacterial, viral, or mycoplasma infection", ok=True)
```

The tool is only active when a `SlakeKG` instance is passed to the dispatcher.

### 5.4 `src/contracts.py` — extend ToolName

Add `"slake_kg"` to the `ToolName` literal.

### 5.5 `scripts/eval_slake.py` — CLI runner

```
Args:
  --data        path to test.json (default: data/Slake1.0/test.json)
  --image-dir   image root (default: data/Slake1.0/imgs)
  --limit       max questions (default: 50, 0 = all)
  --weights     YOLO weights path
  --model       MedGemma model path
  --out         output report path
  --quantize    4-bit quantization
```

### 5.6 Grading — exact match

```python
def exact_match(prediction, ground_truth):
    """Case-insensitive exact match after stripping whitespace."""
    return prediction.strip().lower() == ground_truth.strip().lower()
```

Report structure:
```json
{
  "n": 200,
  "overall_accuracy": 0.45,
  "by_content_type": {
    "KG":          {"n": 40, "accuracy": 0.35},
    "Organ":       {"n": 50, "accuracy": 0.52},
    "Abnormality": {"n": 30, "accuracy": 0.40},
    "Position":    {"n": 35, "accuracy": 0.51},
    "Modality":    {"n": 25, "accuracy": 0.60},
    "Size":        {"n": 10, "accuracy": 0.30},
    "Quantity":    {"n": 10, "accuracy": 0.40}
  },
  "by_answer_type": {
    "OPEN":   {"n": 120, "accuracy": 0.38},
    "CLOSED": {"n": 80,  "accuracy": 0.56}
  },
  "details": [
    {
      "id": "xmlab1_42", "question": "...", "prediction": "...",
      "ground_truth": "...", "score": 1,
      "content_type": "KG", "answer_type": "OPEN",
      "tier": "A", "conf": 0.8, "trace": [...]
    }
  ]
}
```

---

## 6. Testing strategy

### Unit tests (`tests/test_eval_slake.py`)

1. **`test_slake_kg_lookup`** — load a small inline KG, verify
   `lookup("Pneumonia", "cause")` returns expected value.

2. **`test_slake_kg_missing`** — verify `lookup("Unknown", "cause")`
   returns None.

3. **`test_load_slake_filters_xray`** — write a minimal test.json with
   mixed modalities, verify only X-Ray items returned.

4. **`test_exact_match`** — verify case-insensitive matching:
   `"Heart" == "heart"`, `"Yes" != "No"`.

5. **`test_grade_aggregation`** — 4 items with mixed content_type/answer_type,
   verify breakdown math.

6. **`test_cli_smoke`** — mock pipeline + run CLI with `--limit 2`,
   verify JSON report written.

All tests: `pytest tests/test_eval_slake.py -m "not gpu"`.

---

## 7. Boundaries

### Always
- Filter to English + X-Ray only.
- Use `load_slake()` via the common `QAItem` shape.
- Include reasoning trace in report (like VinDr eval).
- Case-insensitive exact match for grading.

### Ask first
- Adding fuzzy/token-level matching (F1 score).
- Extending to MRI/CT modality.
- Modifying OntologyDAG to merge SLAKE KG permanently.

### Never
- Modify existing ontology files (dag.yaml, causal_kg.yaml).
- Hard-code API keys.
- Change the existing VinDr or multihop eval code.
