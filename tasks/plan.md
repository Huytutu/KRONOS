# SLAKE VQA Evaluation — Build Plan

Spec: `eval_slake_SPEC.md`

## Tasks

### Task 1: SlakeKG loader
**File:** `src/knowledge/slake_kg.py`
**What:** Load 3 SLAKE KG CSVs into a dict-of-dicts. `lookup(entity, relation)` returns value or None. `diseases()` and `organs()` list entity names.
**Depends on:** nothing
**Acceptance:** Unit tests verify lookup hit, lookup miss, diseases/organs listing.

### Task 2: SLAKE data loader
**File:** `src/data/loaders.py` (modify)
**What:** Add `load_slake()` returning `List[QAItem]`, filtered by modality and language. Add to LOADERS dict.
**Depends on:** nothing
**Acceptance:** Test verifies X-Ray filtering and QAItem shape.

### Task 3: Register slake_kg tool
**Files:** `src/contracts.py` (add to ToolName), `src/tools/dispatch.py` (route slake_kg)
**What:** Add "slake_kg" to ToolName literal. Route it in dispatch to SlakeKG.lookup().
**Depends on:** Task 1
**Acceptance:** Test verifies dispatch routes slake_kg action correctly.

### Task 4: Eval grading + CLI runner
**Files:** `scripts/eval_slake.py`
**What:** CLI runner with exact-match grading, breakdown by content_type and answer_type, reasoning trace in report.
**Depends on:** Task 1, 2, 3
**Acceptance:** CLI smoke test with mocked pipeline writes valid JSON report.

### Task 5: Final test suite verification
**What:** Run full `pytest tests/ -m "not gpu"` to verify no regressions.
**Depends on:** Task 1-4
**Acceptance:** All tests pass.
