# §3.3 Reflection Loop — Implementation Plan

**Spec:** `graph_reasoning_SPEC.md` §3.3 (verifier-as-critic)
**Goal:** wire the dead `TreeNode.reflection` channel so the deterministic verifier
*teaches* the frozen model: when the model answers but the answer can't be verified,
feed back *why* and let it retry toward Tier A.

---

## Objective

Today, when a model answer verifies to **Tier B** ("answered, unverified") or **ABSTAIN**,
[tree_search.py](src/search/tree_search.py) drops it (`continue`) and the verifier's reason is
lost. After this change, the search re-queues that state **once** with a short reason attached;
the frozen MedGemma reads it via [prompt.py:79-80](src/agent/prompt.py#L79-L80) and gets a second
attempt. MockAgent ignores `reflection`, so existing tests are unaffected.

Decision already taken: **re-queue Tier B once to attempt B→A** (and ABSTAIN likewise).

## Constraints

- Frozen model, deterministic, simple flat code, surgical changes.
- Full suite (currently 319 passed / 6 skipped) must stay green.
- Loop-safe: a state may be reflected **at most once** (`not node.reflection` guard).

## Current hooks (verified in code)

- `TreeNode.reflection: str = ""` — [contracts.py:90](src/contracts.py#L90).
- `build_prompt` already renders `node.reflection` — [prompt.py:79-80](src/agent/prompt.py#L79-L80).
- String-answer branch in `search()` drops B/ABSTAIN via `continue` — the wiring point.
- `verify()` branch structure per qtype — `explain()` will mirror it.

---

## Dependency graph

```
Task 1: explain() in verifier.py  ──►  Task 2: re-queue in tree_search.py
   (pure function + unit tests)          (uses explain(); integration test)
```

Strictly sequential: Task 2 imports `explain` from Task 1. Each task is a complete vertical
slice (code + tests + green suite), not a horizontal layer.

---

## Task 1 — `explain(node, query, dag)` in verifier.py

Add a pure, deterministic function returning a **short, actionable** reason a node failed to
reach Tier A, mirroring the `verify()` failure branches:

- **existential** (no witness, target not a known concept): suggest `re_detect` near a suspected
  region then `is_a`; or `neighbors`/`find_path` to relate the target causally.
- **negation** (exclusion list not fetched): "call `get_exclusion_list('{target}')` first";
  (a listed finding present): "'{target}' relates to a finding that is present".
- **relational** (no localization tool ran): "call `anatomy_of(bbox)` or `compose_laterality(bbox)`
  on the target's bbox".
- **counting / open / unknown**: generic short fallback (these rarely fail to Tier A).

**Acceptance criteria**
- `explain` returns a non-empty string for existential / negation / relational failing nodes.
- The reason references the right next tool for that qtype (assert on a substring).
- Pure: no mutation of inputs; deterministic across runs.

**Verification**
- New `tests/test_verifier.py::test_explain_*` (one per qtype) pass.
- `pytest tests/test_verifier.py -q` green (existing + new).

## Task 2 — re-queue with reflection in tree_search.py

In the **string-answer** proposal branch of `search()`: after `verify(child, ...)`,
if `result.tier in ("B", "ABSTAIN")` **and** `not node.reflection`, append **one** re-queued copy
of the parent `node` with `reflection = explain(child, query, dag)` to the frontier. Use a
single-re-queue guard so multiple failing answers in one expansion don't enqueue duplicates.
Tier-A return and `best_tier_b` capture are unchanged.

**Scope:** only the string-answer branch (the model's explicit answer). The tool-proposal
`reward>=1.0` branch is out of scope for this slice.

**Acceptance criteria**
- A scripted agent that first answers unverifiably **sees a non-empty `node.reflection`** on a
  later expansion (proves the channel is wired and re-queued).
- At most one re-queue per node expansion; no infinite loop (bounded by `budget` + guard).
- MockAgent-driven tests are byte-for-byte unaffected (it ignores `reflection`).

**Verification**
- New `tests/test_tree_search.py::test_reflection_*` pass.
- `pytest -q` → full suite green (≥ 319 passed, 6 skipped), determinism test still passes.

---

## Checkpoints

1. **After Task 1:** `pytest tests/test_verifier.py -q` green → stop, confirm.
2. **After Task 2:** `pytest -q` full suite green → stop, confirm, then commit `§3.3 reflection loop`.

## Risks / mitigations

- *Multiple re-queues per expansion* → single-re-queue flag + `not node.reflection` guard.
- *Reflection never reaching agent* → integration test asserts the agent observed it.
- *Hidden coupling to existing determinism test* → MockAgent ignores reflection; re-run the
  100× determinism test as part of Task 2 verification.

## Out of scope (future)

Tool-proposal-branch reflection; visual-tool reflections; multi-round reflection (>1 retry);
changing MockAgent or the prompt rendering.
