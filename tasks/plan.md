# Fix VinDr VQA ABSTAIN Rate — Build Plan

Spec: `fix_vindr_abstain_SPEC.md`

## Tasks

### Task 1: Add `closure_progress` and `verify` for `open` type
**Files:** `src/engine/verifier.py`, `tests/test_verifier.py`
**What:** Add `_progress_open(node)` returning 0.5 when facts exist, 0.0 otherwise. Update `verify()` open branch: answer + facts → Tier A, answer only → Tier B, empty → ABSTAIN.
**Depends on:** nothing
**Acceptance:**
- `closure_progress` for open + facts → 0.5
- `closure_progress` for open + no facts → 0.0
- `verify` for open + answer + facts → Tier A
- `verify` for open + answer + no facts → Tier B
- `verify` for open + no answer → ABSTAIN
- Existing `test_verify_open_tier_b` updated to match new Tier A behavior

### Task 2: Add `_derive_answer` for `open` type + prompt hints
**Files:** `src/search/tree_search.py`, `src/agent/prompt.py`, `tests/test_tree_search.py`
**What:** Add open branch in `_derive_answer` that joins fact concepts. Add prompt hint for open type ("emit Answer[...] directly"). Add prompt hint for relational with target=None ("use most prominent finding").
**Depends on:** Task 1
**Acceptance:**
- `_derive_answer` for open + facts → comma-joined concepts
- `build_prompt` for open includes direct-answer hint
- `build_prompt` for relational + target=None includes fallback hint
- Tree search for open + facts → non-ABSTAIN result

### Task 3: Switch VinDr judge to Groq
**Files:** `scripts/eval_vindr_vqa.py`
**What:** Replace gemini import with groq import for LLM judge.
**Depends on:** nothing
**Acceptance:**
- `eval_vindr_vqa.py` imports `groq_client.complete`
- No reference to `gemini_client` in eval script

### Task 4: Regression test suite
**What:** Run full `pytest tests/ -m "not gpu"` to verify no regressions.
**Depends on:** Task 1, 2, 3
**Acceptance:** All tests pass.
