# KRONOS Reasoning Redesign — SPEC

Two sequenced parts on the same reasoning engine. Part 1 is a correctness fix to
the existing Tree-of-Thoughts VQA search. Part 2 turns the multi-hop predictor
into a Think-on-Graph loop. Frozen MedGemma, no fine-tuning. Run in `medcxr`.

**Faithfulness ground truth (the load-bearing idea):** an answer is faithful iff
every step of its trace resolves to a *real edge in the KG* and the answer
*collapses* if that chain is removed. The KG is the authority — never the LLM.
This is why the verifier (not the model) decides when reasoning is sufficient.

---

## 1. Objective

Make KRONOS's reasoning both **correct** and **faithful-by-construction**:

- **Part 1 — fix the ToT tree.** The existential verifier currently accepts any
  successful `is_a` as evidence, even an ontology tautology unrelated to the
  image. Bind evidence to detected facts, and give the search value a gradient
  so the tree can actually rank competing branches.
- **Part 2 — ToG-ify the multi-hop predictor.** Replace the one-shot
  propose→verify in `predict_kronos` with a graph-walk: the LLM picks which
  causal edges to explore hop-by-hop, the verifier decides when a path connects
  both findings, and the **path through the KG *is* the trace**.

Target: a workshop/arXiv MVP. Author is an undergraduate — code must read
clearly on the first pass (flat `if`/`elif`, no clever abstractions).

Non-goal: beating the `predict_mock` oracle on accuracy. `common_causes` already
caps shared-cause detection, so Part 2's contribution is **faithful, traceable
reasoning process**, measured by grounding/load-bearing, not by accuracy.

---

## 2. Commands

```bash
conda activate medcxr

# Part 1 — run the verifier/search tests
pytest tests/ -k "verifier or search" -v

# Part 2 — run the multi-hop predictor tests
pytest tests/ -k "multihop or tog" -v

# End-to-end multi-hop eval (existing runner)
python scripts/eval_multihop.py --predictor kronos --limit 50
```

---

## 3. Project structure

Touch only these files. No new top-level modules.

```
src/engine/verifier.py        # Part 1: bind witness to facts, add value gradient
src/search/tree_search.py     # Part 1: _derive_answer existential same binding
src/ontology/dag.py           # Part 2: reuse causal_neighbors/find_causal_path/causal_edge (read-only)
src/eval/predictors.py        # Part 2: rewrite predict_kronos body as the ToG loop
src/eval/multihop_metrics.py  # Part 2: extend _trace_supports to multi-hop chains
tests/                        # new tests for both parts
```

Baselines stay untouched and serve as the comparison table:
`predict_zero_shot`, `predict_cot`, `predict_react` (graph tools, no verifier),
`predict_mock` (oracle).

---

## 4. Part 1 — Fix the ToT tree

### 4.1 Bind the existential witness to detected facts

Today `_has_witness(node)` returns True for *any* successful `is_a` in history.
So `is_a(cardiomegaly, cardiac_abnormality)` — always true on the DAG, unrelated
to the image — makes "Is there Cardiomegaly?" answer **"Yes"** even though the
detected facts are `[Nodule/Mass, Consolidation]`.

Fix: an `is_a` counts as a witness only if its **source node resolves to a
detected fact**.

```python
def _has_witness(node, dag):
    fact_slugs = {dag.resolve_slug(f.concept) for f in node.state_facts}
    return any(
        action.tool == "is_a" and obs.ok and obs.result
        and dag.resolve_slug(action.args.get("node", "")) in fact_slugs
        for action, obs in node.history
    )
```

Apply the same binding in `tree_search._derive_answer` (existential branch):
only return "Yes" when the `is_a` source is a detected fact (or a direct match).
Then the existing closed-world "No" branch fires correctly.

### 4.2 Give `closure_progress` a real gradient

Today the value is effectively `{0.0, 0.1, 0.2, 1.0}` — no in-between — so
best-first never has two distinct branches to rank, and every trace finishes in
one iteration. Negation's `checked/total` is fake (`checked == total` always).

Add a partial-progress term so a node that has checked more facts scores higher
than one that has checked fewer. Keep it simple and monotonic:

- existential: `0.1` base; `+ (facts_checked_via_is_a / total_facts) * 0.7`;
  `1.0` on a bound witness or direct match.
- negation: `0.1` before the exclusion list is fetched; after fetching, score by
  the fraction of exclusion items **actually checked against facts** (real count,
  not `len/len`); `1.0` when all checked and none present.

### 4.3 Part 1 acceptance criteria

1. **Reproduce the bug → fixed:** "Is there Cardiomegaly?" with facts
   `[Nodule/Mass, Consolidation]` returns `answer="No", tier="A"`. → verify with a test.
2. **No regressions:** negation-with-match, counting, closed-world absence,
   relational still produce their current correct tiers. → existing tests pass.
3. **Tree actually branches:** on a case with ≥2 facts and no immediate witness,
   the frontier holds ≥2 nodes with *distinct* rewards at some step. → assert in a test.

---

## 5. Part 2 — ToG loop in `predict_kronos`

### 5.1 The loop (LLM explores, verifier decides)

For a shared-cause question on `(finding_a, finding_b)`:

```
beam = [ path starting at finding_a ]          # topic entity
for hop in range(MAX_DEPTH):                    # MAX_DEPTH = 3
    candidates = []
    for path in beam:
        head = path.head_entity
        neighbors = dag.causal_neighbors(head, "caused_by")   # KG, real edges only
        candidates += [path.extend(n) for n in neighbors]
    # LLM ROLE: prune to top-N most clinically plausible — exploration only
    beam = llm_prune(candidates, finding_a, finding_b, N=BEAM_WIDTH)  # BEAM_WIDTH = 3
    # VERIFIER ROLE: deterministic stop decision — never the LLM
    for path in beam:
        if dag.causal_edge(path.head_entity, finding_b):   # path now links a -> b
            return verified(path)        # answer "Yes", trace = the KG path
return no()                              # depth exhausted -> "No"
```

Rules:
- The LLM **only** ranks/prunes candidate edges. It never says "I have enough."
- The verifier (`causal_edge` / connectivity on the KG) is the **only** thing
  that terminates the search and assigns the answer. This keeps Part 2 consistent
  with Part 1 and with KRONOS's "never trust LLM self-eval" thesis.
- Every edge in the returned trace is a real KG edge by construction — the LLM
  cannot inject a fabricated cause, because candidates come from
  `causal_neighbors`, not from free text.

### 5.2 Trace shape and metric extension

The trace is the **multi-hop chain** the walk produced, e.g.
`[[d1, finding_a], [d2, d1], [d2, finding_b]]`, not necessarily two edges.

Extend `_trace_supports` in `multihop_metrics.py` to validate a chain:
- every edge is a real `causal_edge` (unchanged), **and**
- the edges form a **connected path** that links `finding_a` and `finding_b`
  (replace the current "both names appear as some target" check with an actual
  connectivity check over the trace edges).

`deletion_holds` / `load_bearing_rate` stay as-is — they already test that the
answer collapses when the support edges are removed.

### 5.3 Part 2 acceptance criteria

1. **It walks, not guesses:** `predict_kronos` returns a trace whose edges form a
   connected `finding_a → … → finding_b` chain of real KG edges. → test on a
   known multi-hop item.
2. **Verifier decides termination:** with a stub LLM that always says "enough",
   the loop still only stops when the KG confirms a connecting path. → test.
3. **No fabricated edges:** with a stub LLM that proposes a non-existent cause,
   that cause never appears in the trace. → test.
4. **Grounding metric handles chains:** a 3-hop valid chain scores
   `grounded=True`; a chain with one broken edge scores `grounded=False`. → test.
5. **Baselines untouched:** `predict_react`, `predict_zero_shot`, `predict_cot`,
   `predict_mock` produce the same outputs as before. → regression test / diff.

---

## 6. Code style

- Flat `if`/`elif` chains over abstractions. No metaprogramming, no clever
  one-liners. An undergraduate should read each function once and explain it.
- Plain params with simple defaults (`def llm_prune(candidates, a, b, n=3):`).
  No heavy type annotations unless they aid clarity.
- Match the existing style: pydantic `contracts.py`, `networkx` DAG, small pure
  functions in reading order (main flow first, helpers below).
- Comment only the non-obvious *why* (e.g. "verifier, not LLM, terminates").
- Surgical changes: every changed line traces to Part 1 or Part 2. Do not
  refactor adjacent code or reformat untouched lines.

---

## 7. Testing strategy

- `pytest`, deterministic, no GPU. Stub the LLM prune step with a fake callable
  so Part 2 tests run without MedGemma.
- One test file per part: `tests/test_verifier_witness.py` (Part 1),
  `tests/test_tog_loop.py` (Part 2).
- Each acceptance criterion above maps to exactly one test.
- Reproduce-then-fix for the Part 1 bug: write the failing "Cardiomegaly → No"
  test first, watch it fail, then fix `_has_witness`.

---

## 8. Boundaries

**Always**
- Treat the KG as the single source of truth for evidence and termination.
- Keep models frozen; LLM only proposes/prunes, never decides or verifies.
- Keep the four baseline predictors working as comparison rows.

**Ask first**
- Editing `causal_kg.yaml` / `dag.yaml` content (changes gold answers).
- Changing metric denominators in `grade()` (changes reported numbers).
- Editing `notebook.ipynb` or any committed results file.

**Never**
- Fine-tune any model.
- Let the LLM decide "enough evidence" or self-score a node.
- Accept a trace edge that isn't a real KG edge.
- Delete pre-existing baselines or unrelated dead code.

---

## Sequencing

Part 1 first (correctness — makes every later number trustworthy), then Part 2
(the ToG contribution). Do not start Part 2 until Part 1's tests are green.
