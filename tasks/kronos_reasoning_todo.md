# KRONOS Reasoning Redesign — TODO

Plan: [kronos_reasoning_plan.md](kronos_reasoning_plan.md) · Spec:
[KRONOS_reasoning_SPEC.md](../KRONOS_reasoning_SPEC.md)

Run in `medcxr`. Phase 1 fully green before Phase 2.

## Decisions (confirmed 2026-06-24)
- [x] Ablation remap: `multi_hop=False → max_depth=1`, `reflection=False → prune=False`
- [x] Keep `predict_kronos(item, dag, gen, image=None, …)` signature; new knobs keyword-only

## Phase 1 — Fix the ToT tree
- [x] **T1.1** `_has_witness(node, dag)` binds is_a source to detected facts (verifier.py) — commit 20b86eb
- [x] **T1.2** `_derive_answer` existential binds to detected facts (tree_search.py) — commit 5e43380
- [x] **T1.3** `closure_progress` gradient (existential fraction-checked; negation real count) — commit e48ab6f
- [x] **CHECKPOINT 1**: full non-GPU suite green (357 passed)

## Phase 2 — ToG-ify predict_kronos
- [x] **T2.1** `_trace_supports` validates connected multi-hop chain (multihop_metrics.py) — commit abdf78c
- [x] **T2.2** rewrite `predict_kronos` as KG beam search; verifier-only termination; cause=pivot; trace=full chain; ablations max_depth=1 / prune=False; baselines unchanged — commit 86af222
- [x] **T2.3** runner remap (single_hop→max_depth=1, no_reflection→no_prune); signature kept; CLI green — commit 86af222
- [x] **CHECKPOINT 2**: full non-GPU suite green (364 passed)
