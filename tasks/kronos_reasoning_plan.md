# KRONOS Reasoning Redesign — Implementation Plan

Spec: [KRONOS_reasoning_SPEC.md](../KRONOS_reasoning_SPEC.md)

Two phases, sequenced. **Phase 1 must be fully green before Phase 2 starts.**
Each task is one vertical slice: failing test → minimal change → green.

Run everything in the `medcxr` conda env. `pytest` is deterministic, no GPU
(the LLM is always stubbed).

---

## Dependency graph

```
Phase 1 (correctness — independent of Phase 2)
  T1.1 verifier witness binding ──┐
  T1.2 derive-answer binding   ───┼──> CHECKPOINT 1 (verifier + tree_search green)
  T1.3 closure_progress gradient ─┘

Phase 2 (ToG — depends on Phase 1 being green)
  T2.1 metric: multi-hop trace support  (pure, safe first slice)
        │
  T2.2 predict_kronos -> ToG beam loop  (depends on T2.1 for grading)
        │
  T2.3 signature/runner compatibility   (notebook + eval CLI still work)
        └──> CHECKPOINT 2 (multihop + CLI green, baselines untouched)
```

T1.1 and T1.2 are the same idea (bind `is_a` evidence to detected facts) at two
layers; T1.3 is independent and can land in parallel. In Phase 2, T2.1 is the
safe pure-function slice and is done first so T2.2 has a working grader.

---

## Known conflict to resolve (flagged for review)

`predict_kronos` currently has **5 tests** pinning its propose→verify contract,
including the ablation kwargs `multi_hop=False` and `reflection=False`
([test_multihop_predict.py:118-146](../tests/test_multihop_predict.py#L118-L146)).
Rewriting its body into a ToG loop **breaks all 5**. These are NOT baselines
(the baselines are `zero_shot`/`cot`/`react`/`mock`, which stay untouched), so
rewriting them is expected — but two decisions must be confirmed before T2.2:

1. **Ablation remap.** The old ablations (`multi_hop`, `reflection`) are listed
   as paper ablations. Proposed ToG equivalents:
   - `multi_hop=False`  → `max_depth=1` (single-hop shared-cause only)
   - `reflection=False` → `prune=False` (no LLM pruning; explore full beam)
   Confirm this mapping, or drop the ablations.
2. **Signature stays compatible:** `predict_kronos(item, dag, gen, image=None)`
   so the notebook and `scripts/eval_multihop.py` keep calling it unchanged.
   New ToG knobs (`beam_width`, `max_depth`, `prune`) are keyword-only with
   defaults.

---

## Phase 1 — Fix the ToT VQA tree

### T1.1 — Bind existential witness to detected facts (verifier)

**Change:** `src/engine/verifier.py`
- `_has_witness(node)` → `_has_witness(node, dag)`: an `is_a` counts only if
  `dag.resolve_slug(action.args["node"])` is in the detected-fact slugs.
- Pass `dag` at both call sites (`_progress_existential`, `_verify_existential`).

**Reproduce-then-fix test** (`tests/test_verifier.py`, new):
- facts `[Nodule/Mass, Consolidation]`, history `is_a(cardiomegaly,
  cardiac_abnormality)=ok`, target `Cardiomegaly`, answer `Yes`.
- Before fix: `verify` returns tier A "Yes" (bug). After fix: tier A **"No"**
  (closed-world absence fires because no detected fact is-a Cardiomegaly).

**Acceptance**
- New test passes (was failing).
- All existing `test_verifier.py` pass unchanged — verified safe because their
  witnesses have source == a detected fact (e.g. fact "Cardiomegaly" +
  `is_a(cardiomegaly, …)`).

**Verify:** `pytest tests/test_verifier.py -v`

---

### T1.2 — Bind `_derive_answer` existential to detected facts (search)

**Change:** `src/search/tree_search.py`
- In `_derive_answer` existential branch, return "Yes" only when the successful
  `is_a` source resolves to a detected fact (mirror T1.1); otherwise fall through
  to the direct-match check, else "No".

**Test** (`tests/test_tree_search.py`, new): with a stub agent that proposes
`is_a(cardiomegaly, cardiac_abnormality)` on facts `[Nodule/Mass, Consolidation]`,
the search result is "No"/tier A, not "Yes".

**Acceptance**
- New test passes; existing `test_tree_search.py` pass unchanged.

**Verify:** `pytest tests/test_tree_search.py -v`

---

### T1.3 — Real gradient in `closure_progress`

**Change:** `src/engine/verifier.py`
- existential: `0.1` base `+ (facts_checked_via_is_a / total_facts) * 0.7`;
  `1.0` on a bound witness or direct match. "Checked" = the fact's slug was the
  source of an `is_a` action (ok or not).
- negation: after the exclusion list is fetched, score by the **real fraction of
  exclusion items checked against facts**, not `len/len`. `1.0` when all checked
  and none present; `0.0` if any present.

**Test** (`tests/test_verifier.py` + `tests/test_tree_search.py`, new):
- A branching case (≥2 facts, no immediate witness) where the frontier holds
  ≥2 nodes with **distinct** rewards at some step.
- existing `test_progress_*` still satisfy their bounds (they assert ranges like
  `0 < p < 1`, not exact mid-values — verified compatible).

**Acceptance**
- New gradient test passes; existing progress tests pass.

**Verify:** `pytest tests/test_verifier.py tests/test_tree_search.py -v`

---

## CHECKPOINT 1 — Phase 1 gate

```bash
pytest tests/test_verifier.py tests/test_tree_search.py -v
```
All green, including the "Cardiomegaly → No" reproduce test. **Do not start
Phase 2 until this passes.** Commit Phase 1 as its own change.

---

## Phase 2 — ToG-ify `predict_kronos`

### T2.1 — Multi-hop trace support in the metric (pure, safe first)

**Change:** `src/eval/multihop_metrics.py`
- Extend `_trace_supports`: every edge is a real `causal_edge` (unchanged) **and**
  the trace edges form a **connected path** linking `finding_a` and `finding_b`
  (replace the "both names appear as some target" check with real connectivity).
- Keep the existing 2-edge shared-cause shape valid (it is a connected path of
  length 2), so `predict_mock` grading stays at `grounding_rate == 1.0`.

**Tests** (`tests/test_multihop_metrics.py`, new):
- Valid 3-hop chain `a → d1 → d2 → b` (all real edges) → `grounded=True`.
- Same chain with one fabricated edge → `grounded=False`.
- The old 2-edge oracle trace → still `grounded=True`.

**Acceptance**
- New tests pass; existing `test_multihop_metrics.py` and the `predict_mock`
  grading test (`test_write_predictions_schema_and_grade`) pass unchanged.

**Verify:** `pytest tests/test_multihop_metrics.py tests/test_multihop_predict.py -k mock -v`

---

### T2.2 — Rewrite `predict_kronos` as a ToG beam search

**Change:** `src/eval/predictors.py`
- Body = beam search on the causal KG:
  - beam starts at `finding_a`; each hop expands via
    `dag.causal_neighbors(head, "caused_by")`.
  - `llm_prune(candidates, a, b, gen, n)` ranks/keeps top-N — **exploration
    only**; never decides termination. Define a small, stubbable contract
    (input: candidate next-nodes; output: ordered subset).
  - The verifier — `dag.causal_edge(head, finding_b)` / path connectivity — is
    the **only** thing that terminates and assigns the answer.
  - `answer="Yes"`, `cause = pivot disorder on the path`, `trace = full KG chain`.
    "No" when depth exhausted with no connecting path.
- Defaults: `beam_width=3`, `max_depth=3`, `prune=True`. Ablations per the remap
  decision above.

**Rewrite the 5 existing `predict_kronos` tests** to the ToG contract
(`tests/test_multihop_predict.py`):
- connected-chain trace on a known item;
- **verifier decides termination**: a stub LLM that always says "enough" still
  only stops when the KG confirms a connecting path;
- **no fabricated edge**: a stub proposing a non-existent cause never appears in
  the trace;
- ablations: `max_depth=1` limits to single-hop; `prune=False` explores full beam.

**Acceptance**
- New ToG tests pass.
- Baselines untouched: `predict_zero_shot`, `predict_cot`, `predict_react`,
  `predict_mock` produce identical outputs (their tests pass unchanged).

**Verify:** `pytest tests/test_multihop_predict.py -v`

---

### T2.3 — Signature & runner compatibility

**Change:** none beyond keeping `predict_kronos(item, dag, gen, image=None, …)`.
- Confirm `scripts/eval_multihop.py` and the notebook predictor list still call
  it without edits.

**Acceptance**
- `test_eval_multihop_cli.py` and `test_run_multihop_cli.py` pass unchanged.

**Verify:** `pytest tests/test_eval_multihop_cli.py tests/test_run_multihop_cli.py -v`

---

## CHECKPOINT 2 — Phase 2 gate

```bash
pytest tests/ -v        # full suite green
```
ToG loop produces connected multi-hop traces; verifier (not the LLM) terminates;
baselines unchanged; CLI/notebook compatible. Commit Phase 2 as its own change.

---

## Boundaries (from spec §8)

- Never edit `causal_kg.yaml` / `dag.yaml` content, `grade()` denominators, or
  `notebook.ipynb` without asking.
- LLM proposes/prunes only; the KG verifier decides and terminates.
- Surgical changes; flat readable code; every changed line traces to a task.
