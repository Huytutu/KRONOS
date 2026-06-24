# KRONOS Reasoning Redesign — TODO

Plan: [kronos_reasoning_plan.md](kronos_reasoning_plan.md) · Spec:
[KRONOS_reasoning_SPEC.md](../KRONOS_reasoning_SPEC.md)

Run in `medcxr`. Phase 1 fully green before Phase 2.

## Decisions (confirmed 2026-06-24)
- [x] Ablation remap: `multi_hop=False → max_depth=1`, `reflection=False → prune=False`
- [x] Keep `predict_kronos(item, dag, gen, image=None, …)` signature; new knobs keyword-only

## Phase 1 — Fix the ToT tree
- [ ] **T1.1** `_has_witness(node, dag)` binds is_a source to detected facts (verifier.py); pass `dag` at both call sites
  - [ ] Reproduce test: facts [Nodule/Mass, Consolidation] + is_a(cardiomegaly,…) → verify "No"/A
  - [ ] `pytest tests/test_verifier.py` green
- [ ] **T1.2** `_derive_answer` existential binds to detected facts (tree_search.py)
  - [ ] Stub-agent search test → "No"/A, not "Yes"
  - [ ] `pytest tests/test_tree_search.py` green
- [ ] **T1.3** `closure_progress` gradient (existential fraction-checked; negation real count)
  - [ ] Branching test: frontier ≥2 nodes with distinct rewards
  - [ ] existing `test_progress_*` still pass
- [ ] **CHECKPOINT 1**: `pytest tests/test_verifier.py tests/test_tree_search.py` all green → commit

## Phase 2 — ToG-ify predict_kronos
- [ ] **T2.1** `_trace_supports` validates connected multi-hop chain (multihop_metrics.py)
  - [ ] tests: 3-hop valid True, broken-edge False, 2-edge oracle still True
  - [ ] `predict_mock` grading still `grounding_rate==1.0`
- [ ] **T2.2** rewrite `predict_kronos` body as KG beam search (predictors.py)
  - [ ] `llm_prune` stubbable contract; verifier-only termination; cause=pivot disorder; trace=full chain
  - [ ] rewrite 5 old predict_kronos tests to ToG contract (incl. "LLM says enough" + "no fabricated edge")
  - [ ] ablations: max_depth=1, prune=False
  - [ ] baselines (zero_shot/cot/react/mock) outputs unchanged
- [ ] **T2.3** signature/runner compatibility
  - [ ] `test_eval_multihop_cli.py` + `test_run_multihop_cli.py` green
- [ ] **CHECKPOINT 2**: `pytest tests/` full suite green → commit
