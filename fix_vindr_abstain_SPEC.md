# Fix VinDr VQA ABSTAIN Rate — Specification

## Objective

Reduce the ABSTAIN rate on VinDr-CXR VQA evaluation from ~60% to <15% and fix
the LLM judge so correctly-answered questions actually score points.

**Target**: VinDr-CXR VQA pipeline only. SLAKE is out of scope for this spec.

## Problem Analysis

Six VinDr VQA results showed 0% accuracy across 10 questions. Two root causes:

### P1. Gemini judge broken (all scores = 0)
`eval_vindr_vqa.py` uses `gemini_client.complete` as LLM-as-judge, but the
Gemini free-tier quota is exhausted → judge returns `""` → every score = 0, even
for correct predictions like `"Yes"` for `"Is there Cardiomegaly?"`.

### P2. Pipeline ABSTAINs on 6/10 questions
Three distinct bugs cause ABSTAIN:

**P2a. `open` type: `closure_progress` returns 0.0**
`verifier.closure_progress` has no case for `qtype == "open"` → falls through to
`return 0.0` (line 26). This means:
- Root node gets `reward = 0.0`
- Agent is called once, but if it proposes tool actions instead of `Answer[...]`,
  the child node also gets `reward = 0.0` → fails the `child.reward > 0` gate
  (tree_search.py:96) → child is dropped
- Search ends with ABSTAIN

**P2b. `open` verifier caps at Tier B**
`verify()` for `qtype == "open"` always returns Tier B (verifier.py:33-37). This
is acceptable when the answer is non-empty (best_tier_b captures it), but the
combination of P2a (no exploration) means the agent rarely produces an answer.

**P2c. `relational` with `target=None`**
Questions like "Which side shows the abnormality?" parse as `relational` with
`target=None`. The agent prompt says "Find the fact matching the target" but
there is no target, so the agent doesn't know which fact's bbox to pass to
`anatomy_of`/`compose_laterality`. The verifier then finds no qualifying action
in history → ABSTAIN.

## Changes

### C1. Switch VinDr judge from Gemini to Groq
**File**: `scripts/eval_vindr_vqa.py`

Replace `from src.llm.gemini_client import complete as gemini_complete` with
`from src.llm.groq_client import complete as groq_complete` and pass it to
`grade_batch`. The function signature is identical: `complete(prompt) → str`.

**Acceptance**: `groq_complete("Reply with exactly CORRECT")` returns a string
containing "CORRECT".

### C2. Add `closure_progress` for `open` type
**File**: `src/engine/verifier.py`, function `closure_progress`

Add a case before the final `return 0.0`:
```
if qtype == "open":
    return _progress_open(node)
```

`_progress_open(node)` returns:
- `0.5` if the node has detected facts (facts alone are partial evidence)
- `0.0` if no facts at all (nothing to summarize)

This gives the root node a non-zero reward so tool-calling children pass the
`child.reward > 0` gate and the agent gets multiple chances to answer.

**Acceptance**: A root node with facts gets `reward = 0.5`, children stay in
the frontier.

### C3. Promote `open` answers to Tier A when grounded in facts
**File**: `src/engine/verifier.py`, function `verify`

Change the `open` branch: if the node has a non-empty answer AND at least one
detected fact, return Tier A (the answer is grounded). If the node has a
non-empty answer but no facts, keep Tier B (ungrounded guess). If answer is
empty, ABSTAIN.

```
if qtype == "open":
    if node.answer:
        tier = "A" if node.state_facts else "B"
        return SearchResult(answer=node.answer, tier=tier, path=path, conf=_min_conf(node))
    return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)
```

**Acceptance**: `open` question with answer + facts → Tier A. Without facts →
Tier B. Empty answer → ABSTAIN.

### C4. Auto-resolve `relational` target from facts
**File**: `src/search/tree_search.py`, function `_derive_answer`

In the `relational` branch, when no `anatomy_of`/`compose_laterality` result is
in history but facts exist, fall back to calling the first fact's bbox info:

No change to `_derive_answer` itself — the fix goes into the prompt and the
`closure_progress` for relational.

**File**: `src/agent/prompt.py`, function `build_prompt`

When `query.type == "relational"` and `query.target is None`, append a hint:
```
Target: (not specified — use the most prominent finding from evidence facts)
```
And list each fact's bbox so the agent knows what to pass to `anatomy_of`.

**File**: `src/engine/verifier.py`, function `_progress_relational`

Currently returns 0.2 when no anatomy/laterality action exists. Keep this —
it already allows exploration. No change needed.

**Acceptance**: "Which side shows the abnormality?" with detected facts →
agent proposes `anatomy_of(bbox)` or `compose_laterality(bbox)` using the
first/most-confident fact's bbox → verifier returns Tier A.

### C5. Add `_derive_answer` for `open` type
**File**: `src/search/tree_search.py`, function `_derive_answer`

The current fallback at the end is `return ""`. Add an `open` branch before it
that summarizes detected facts:

```
if qtype == "open":
    if node.state_facts:
        return ", ".join(sorted({f.concept for f in node.state_facts}))
    return ""
```

This means when `closure_progress` reaches 1.0 (which it won't for open, since
max is 0.5, but if reward >= 1.0 triggers), or when the agent emits Answer[...]
the answer is non-empty.

Actually, since open's closure_progress caps at 0.5, `_derive_answer` is only
called when `child.reward >= 1.0` — which won't happen for open. So this
branch is only a safety net. The real answer comes from the agent's
`Answer[...]` emission.

**Acceptance**: `_derive_answer` for open + facts returns a non-empty string.

### C6. Improve agent prompt for `open` questions
**File**: `src/agent/prompt.py`, function `build_prompt`

When `query.type == "open"` and there are detected facts, append a stronger
hint after the facts section:

```
You have detected findings listed above. Summarize them to answer the question.
Emit Answer[your answer] directly — no tool calls needed.
```

This steers MedGemma toward emitting `Answer[...]` on the first call, avoiding
the dropped-child problem entirely.

**Acceptance**: For an open question with facts, the agent's first response
is `Answer[...]` with a non-empty summary.

## Files Changed

| File | Change |
|---|---|
| `scripts/eval_vindr_vqa.py` | C1: swap gemini → groq import |
| `src/engine/verifier.py` | C2: add `_progress_open`; C3: upgrade open verify |
| `src/search/tree_search.py` | C5: add open branch in `_derive_answer` |
| `src/agent/prompt.py` | C4: relational no-target hint; C6: open direct-answer hint |

## Files NOT Changed

| File | Reason |
|---|---|
| `src/llm/gemini_client.py` | Already fixed (`load_dotenv()`) earlier this session |
| `src/llm/groq_client.py` | No changes needed, same interface |
| `src/contracts.py` | No schema changes |
| `src/question/parser.py` | Parsing works correctly for VinDr questions |
| `src/tools/*` | Tool implementations are correct |

## Testing Strategy

- **Unit tests**: Extend `tests/test_verifier.py` to cover `open` type in both
  `closure_progress` and `verify`.
- **Unit tests**: Extend `tests/test_tree_search.py` to verify open-type search
  produces a non-ABSTAIN result when agent returns Answer[...].
- **Integration**: Run `eval_vindr_vqa.py --limit 10` and verify ABSTAIN count
  drops from 6 to ≤1.
- **Regression**: `pytest tests/ -m "not gpu"` must pass.

## Boundaries

- **Always**: Keep changes surgical — only touch the files listed above.
- **Always**: Preserve existing Tier A/B/ABSTAIN semantics for other question
  types (existential, negation, counting, shared_cause).
- **Never**: Change the YOLO detector or MedGemma model weights.
- **Never**: Add new question types or tools in this change.
- **Ask first**: If `relational` with `target=None` needs a parser-level fix
  (auto-filling target from facts before search), check with user.
