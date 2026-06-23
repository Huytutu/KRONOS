# SPEC — Tier B for Unverified Agent Answers

> Scope: change the verifier so it **never blocks** a direct answer from
> MedGemma. When the ontology can verify → tier A (unchanged). When it
> can't → **tier B** (advisory) instead of ABSTAIN. ABSTAIN only when
> no answer exists at all.
>
> **Why:** the current verifier treats the ontology as a gate — if the DAG
> trace isn't perfect, the answer is thrown away. This silently discards
> correct MedGemma answers for Where/Which/negation questions. The ontology
> should **upgrade** answers, not block them.

---

## 1. The problem

Current verify() for 3 question types:

```
_verify_relational:  DAG trace (anatomy_of) exists? → A. Else → ABSTAIN.
_verify_negation:    exclusion list fetched?         → A. Else → ABSTAIN.
_verify_existential: witness or closed-world?        → A. Else → ABSTAIN.
```

MedGemma often proposes `Answer["left lung"]` or `Answer["Yes"]` directly
without the tool trace. The verifier discards these. Result: 64% ABSTAIN
rate on VinDr-CXR-VQA.

## 2. The fix

One rule: **if the node has a non-empty answer, never return ABSTAIN.**

```
_verify_relational:  DAG trace exists? → A. Has answer? → B. Else → ABSTAIN.
_verify_negation:    exclusion list ok? → A. Has answer? → B. Else → ABSTAIN.
_verify_existential: witness/closed-world? → A. Has answer? → B. Else → ABSTAIN.
```

The tree search already handles tier B correctly (lines 46-47 in
tree_search.py: `if result.tier == "B" and best_tier_b is None`).
After budget exhaustion it returns `best_tier_b` if one exists.

## 3. What changes

**File: `src/engine/verifier.py`** — 3 functions, same pattern each:

### `_verify_existential` (line 60)

Before:
```python
return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)
```

After:
```python
if node.answer:
    return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)
```

### `_verify_negation` (line 98)

Before (line 103, when excl_list is None):
```python
return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)
```

After:
```python
if node.answer:
    return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)
```

### `_verify_relational` (line 125)

Before (line 133):
```python
return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)
```

After:
```python
if node.answer:
    return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)
```

## 4. What does NOT change

- **Tier A logic** — identical. Ontology verification is unchanged.
- **closure_progress** — unchanged. Search still prefers nodes with DAG progress.
- **Tree search** — unchanged. Already handles tier B.
- **Counting / open** — unchanged. Counting already gets A; open already gets B.
- **conf for tier B** — set to `0.3` (lower than tier A's `_min_conf`), so
  the eval can distinguish verified vs unverified answers.

## 5. Expected impact

| Metric | Before | After |
|---|---|---|
| Tier A | ~35% | ~35% (unchanged) |
| Tier B | ~1% | ~30% (MedGemma answers that were ABSTAIN) |
| ABSTAIN | ~64% | ~35% (only when MedGemma has no answer) |
| Coverage | ~36% | ~65% |

Tier A accuracy stays the same (verified). Overall accuracy may drop slightly
because tier B includes unverified answers, but coverage doubles.

## 6. Testing

| Test | What it proves |
|---|---|
| Existing tier-A tests | Unchanged — all must still pass |
| New: existential with answer but no witness → tier B | Agent answer preserved |
| New: relational with answer but no anatomy tool → tier B | Agent answer preserved |
| New: negation with answer but no excl list → tier B | Agent answer preserved |
| New: no answer at all → still ABSTAIN | ABSTAIN only when genuinely empty |
| Faithfulness: retrieve-only node still can't reach A | Tier A integrity preserved |

## 7. Boundaries

**Always:**
- Tier A logic is untouched — never weaken the verified tier
- Tier B conf is lower than any tier A conf

**Never:**
- Promote an unverified answer to tier A
- Return tier B when node.answer is empty/None

**Ask first:**
- Whether eval metrics should report tier A and tier B accuracy separately
