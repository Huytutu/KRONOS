# SPEC — Symbolic Engine (`src/engine/engine.py`)

> References: [README §5.0-5.4](docs/README.md), [project.tex §Bước 6](docs/project.tex),
> [IMPLEMENTATION §5.2](docs/IMPLEMENTATION.md)

---

## 1. Objective

Deterministic, query-typed graph-reasoning over `K = (facts ∪ ontology)`.

```
Engine(Q, K) → ⟨verdict, derivation⟩
    verdict ∈ {PASS, FAIL, UNVERIFIABLE}
    derivation = list of reasoning steps (the faithful trace)
```

Same code, two roles:
- **E_sym proposer:** `engine.run(query, facts)` → find an answer + emit Candidate.
- **Verifier:** `engine.verify(candidate, query, facts)` → check a candidate.

The derivation is a **byproduct** of the search — not a post-hoc explanation.
Same K + same Q → same output (deterministic, no randomness, no weights).

---

## 2. Verdict type (add to `contracts.py`)

```python
Verdict = Literal["PASS", "FAIL", "UNVERIFIABLE"]

class EngineResult(BaseModel):
    verdict: Verdict
    answer: str                          # e.g. "Yes", "left", "3"
    derivation: list                     # reasoning steps used
    conf: float                          # min fact confidence in derivation
```

---

## 3. Logic per question type (closure conditions)

### existential — "Is there X?"
1. Resolve `query.target` (canonical name) → slug via `dag.get_node_by_name(target)`.
2. For each fact: resolve `fact.concept` → slug via `dag.get_node_by_name`.
   Then check `dag.reachable_is_a(fact_slug, target_slug)` (returns the path, or
   None). Exact match is the trivial path `[slug]` when fact_slug == target_slug.
3. **First witness** (path found) → PASS (early-exit, monotone). The returned
   path IS the derivation.
4. No witness → FAIL.
5. Target not resolvable to a DAG node → UNVERIFIABLE.

Derivation uses the real `reachable_is_a` path:
`["Consolidation observed (conf=0.88)", "consolidation → airspace_abnormality → pulmonary_abnormality", "PASS"]`
(note: 2 hops, not 1 — the trace must be the actual path, for faithfulness).

**Key difference from E_perc:** E_perc only checks exact name-match.
Engine uses `is-a*` transitive — so "Consolidation" satisfies "Is there pulmonary abnormality?"

### negation — "Is there no X?"
1. Get exclusion list `E(target)` via `dag.get_exclusion_list(target)`.
   This returns a list of **slugs** (e.g. `["pneumothorax", "pleural_effusion"]`).
2. **No exclusion list** (None) → UNVERIFIABLE (abstain — cannot guarantee completeness).
3. Build the set of observed slugs: `{dag.get_node_by_name(f.concept) for f in facts}`.
4. For each slug in E(target): check if it (or any is-a descendant) is observed.
5. **Any match found** → FAIL (X is present, cannot say absent), conf = that fact's conf.
6. **All checked, none found** → PASS (absent under closed-world), conf = 1.0.

Derivation: `["Checking exclusion list for Pneumothorax", "pneumothorax: not observed", "pleural_effusion: not observed", "PASS (absent)"]`

**Name↔slug note:** facts carry canonical names ("Pleural effusion"); exclusion
lists and DAG nodes use slugs ("pleural_effusion"). Always bridge via
`dag.get_node_by_name` before comparing.

### relational — "Where is X?" / "Which side?"
1. Find fact matching target (exact or is-a).
2. If `constraints.attr == "laterality"` → return `fact.laterality` directly
   (already computed by perception — no DAG call needed).
3. If `constraints.attr == "location"` → `dag.anatomy_of(fact.bbox, 512, 512)`
   → anatomy zone name. (Facts are at 512x512 per the perception spec.)
4. Fact not found → FAIL.
5. Cannot resolve attribute (e.g. anatomy_of returns None) → UNVERIFIABLE.

### counting — "How many findings?"
1. Count distinct observed concepts in facts.
2. Always PASS with the count as answer.

### open — "What abnormality is visible?"
1. List all observed concepts from facts.
2. Always PASS with comma-joined list.

---

## 4. Module interface

File: `src/engine/engine.py`

```python
class Engine:
    def __init__(self, dag):
        # dag: OntologyDAG instance

    def run(self, query, facts):
        # query: Query
        # facts: list[PerceptualFact]
        # returns: EngineResult

    def propose(self, query, facts):
        # Convenience: run() then wrap as Candidate with head_id="E_sym"
        # returns: Candidate or None

    def verify(self, candidate, query, facts):
        # Check if candidate's answer is consistent with K.
        # returns: EngineResult
```

Dependencies: `OntologyDAG`, `contracts.*`. No dependency on parser, linker,
perception, or other proposers.

---

## 5. Concept resolution: canonical name → DAG node

Facts use canonical names ("Cardiomegaly"). DAG uses slugs ("cardiomegaly").
Engine bridges via `dag.get_node_by_name(fact.concept)`.

If a fact's concept cannot be resolved to a DAG node, that fact is **skipped**
(not an error — it means the finding is outside ontology coverage).

---

## 6. Testing

### Unit tests (`tests/test_engine.py`)

**Existential:**
- Target in facts (exact match) → PASS.
- Target NOT in facts but a child is (is-a reasoning) → PASS.
  E.g., facts=[Consolidation], query target="Pulmonary abnormality" → PASS
  via `consolidation is-a airspace_abnormality is-a pulmonary_abnormality`.
- Target not in facts, no child → FAIL.
- Unknown target → UNVERIFIABLE.

**Negation:**
- Target in facts → FAIL.
- Target not in facts, exclusion list fully checked → PASS.
- No exclusion list for target → UNVERIFIABLE.
- Exclusion list partner found in facts → FAIL.
  E.g., target="Consolidation", facts=[Infiltration] → FAIL
  (infiltration is in Consolidation's exclusion list).

**Relational:**
- Laterality query, fact found → PASS with laterality.
- Location query, fact found → PASS with anatomy zone.
- Fact not found → FAIL.

**Counting:**
- 3 findings → PASS, answer="3".
- Empty facts → PASS, answer="0".

**Determinism:**
- Same (Q, K) → identical result across 100 runs.

**Verify mode:**
- candidate says "Yes" + engine confirms → PASS.
- candidate says "Yes" + engine contradicts → FAIL.

---

## 7. Boundaries

### Always do
- Use `is-a*` transitive reachability, not just exact match (the whole point).
- Return UNVERIFIABLE rather than guess when coverage is insufficient.
- Include derivation steps (the faithful trace) in every result.
- Be deterministic: same inputs → same output.

### Never do
- Use learned weights, sampling, or any non-deterministic operation.
- Guess when exclusion list is missing (return UNVERIFIABLE).
- Depend on parser, linker, perception, or other proposers.
- Modify the DAG or facts during reasoning.
