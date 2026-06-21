# Plan — v4 Deterministic Core

Source: [v4_core_SPEC.md](../v4_core_SPEC.md)

## Task 1: Update contracts — add v4 types, remove Candidate

Add `Tier`, `ToolName`, `Action`, `Observation`, `TreeNode`, `SearchResult` to
`src/contracts.py`. Remove `Candidate` (v2 multi-head type). Add `QType` value
`"open"` for Tier-B questions.

**Depends on:** nothing  
**Accept:** `Action(tool="is_a", args={"node":"x","target":"y"})` builds;
`TreeNode` holds history of `(Action, Observation)` pairs; `SearchResult` has
`tier ∈ {"A","B","ABSTAIN"}`. Existing tests still pass (no regressions in
parser/linker/dag tests).

---

## Task 2: Symbolic tool layer — `src/tools/symbolic.py`

Create `run_tool(action, facts, dag, img_wh) → Observation` that dispatches
on `action.tool` to the matching `OntologyDAG` method. Pure function.

**Depends on:** Task 1 (Action, Observation types)  
**Accept:** `run_tool(Action(tool="is_a", args={"node":"cardiomegaly","target":"cardiac_abnormality"}), ..., dag, ...)` returns `Observation(result=["cardiomegaly","cardiac_abnormality"], ok=True)`. All 6 tools dispatch correctly. Unit tests in `tests/test_symbolic_tools.py`.

---

## Task 3: Verifier — `src/engine/verifier.py`

Implement `closure_progress(node, query, dag) → float` and
`verify(node, query, dag) → SearchResult`.

**Depends on:** Task 1 (TreeNode, SearchResult), Task 2 (run_tool for replay)  
**Accept:**
- existential: witness found → progress 1.0, verify → Tier A.
- negation: all exclusions absent → Tier A; one present → conf=0; missing list → ABSTAIN.
- relational: anatomy resolved → Tier A.
- counting: count matches → Tier A.
- disjoint violation in history → reward 0.
- Unit tests in `tests/test_verifier.py`.

---

## Task 4: Agent interface + MockAgent

Create `src/agent/base.py` with `Agent` Protocol (`propose_actions(node, query, k) → list`).
Create `src/agent/mock.py` with `MockAgent` returning scripted actions per question type.

**Depends on:** Task 1 (Action, TreeNode, Query)  
**Accept:** `MockAgent().propose_actions(node, query, 3)` returns a list of
`Action` objects appropriate for the query type. Unit tests in `tests/test_mock_agent.py`.

---

## Task 5: Tree search — `src/search/tree_search.py`

Implement `search(query, facts, dag, agent, budget, k) → SearchResult`.
Best-first: expand highest-reward node, run tools, evaluate with verifier,
backtrack implicitly via frontier.

**Depends on:** Task 2, Task 3, Task 4  
**Accept:**
- Existential 2-hop: finds witness via MockAgent, returns Tier A, path matches
  `reachable_is_a` path.
- Backtrack: dead branch first, valid branch second → still Tier A.
- Budget exhausted → ABSTAIN.
- Determinism: same inputs → identical result 100 runs.
- Deletion test: remove witness fact → answer flips.
- Unit tests in `tests/test_tree_search.py`.
