# §3.3 Reflection Loop — Task List

Source: `tasks/reflection_loop_plan.md`. Build with `/build auto` after approval.

## Task 1 — explain() in verifier.py  [depends: none]
- [ ] Add `explain(node, query, dag)` mirroring `verify()` failure branches per qtype.
- [ ] Returns short actionable reason (existential / negation / relational); generic fallback otherwise.
- [ ] Pure + deterministic; no input mutation.
- [ ] Tests: `test_explain_existential`, `test_explain_negation`, `test_explain_relational` (assert non-empty + right next-tool substring).
- [ ] **Verify:** `pytest tests/test_verifier.py -q` green.
- [ ] **Checkpoint 1:** stop, confirm.

## Task 2 — re-queue with reflection in tree_search.py  [depends: Task 1]
- [ ] In the string-answer branch: on Tier B/ABSTAIN and `not node.reflection`, append one re-queued `node` copy with `reflection = explain(child, query, dag)`.
- [ ] Single-re-queue guard per expansion; Tier-A return + `best_tier_b` unchanged.
- [ ] Test: scripted agent observes a non-empty `node.reflection` after answering unverifiably.
- [ ] Test: existing MockAgent tests + determinism (100×) unaffected.
- [ ] **Verify:** `pytest -q` full suite green (≥ 319 passed, 6 skipped).
- [ ] **Checkpoint 2:** stop, confirm, then commit `feat(search): §3.3 reflection loop`.
