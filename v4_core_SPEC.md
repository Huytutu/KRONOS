# SPEC — KRONOS v4 Deterministic Core (Verifier-Guided Tree Search)

> Scope: the **deterministic core only** — Steps 1–4 of the v4 migration.
> Builds: contract types, the verifier (verify-gate + closure-progress), the
> symbolic tool layer, and the tree search with a **mock agent**.
>
> **Out of this spec (a later one):** LLaVA-Med agent, the 3 visual tools
> (`inspect`/`re_detect`/`compare`), EvaX swap. Perception stays **YOLO** for now.
>
> References: [intent](docs/intent/kronos-v4-multimodal-tree-search.md),
> [ontology_SPEC.md](ontology_SPEC.md) (DAG, already built).
> Supersedes the v2 propose-then-verify design (engine spec removed).

---

## 1. Objective

Build a **deterministic, testable, GPU-free backbone** for KRONOS v4: a best-first
tree search over symbolic graph operations, guided by a deterministic verifier
(verifier-as-value), that returns an answer + the winning root→leaf path as a
faithful trace.

```
TreeSearch(query, evidence_graph, agent) → ⟨answer, tier, path⟩
    tier ∈ {A, B, ABSTAIN}
    path = list of (action, observation) — the faithful trace (Tier A)
```

The `agent` is an **interface**: a mock implements it now (scripted actions for
tests); LLaVA-Med implements the same interface later. The search, verifier, and
tools are deterministic — same inputs → same output.

**Why deterministic-first:** the faithfulness guarantee lives entirely in the
symbolic part. We build and test it with zero neural dependency, then plug in the
agent and visual tools behind fixed interfaces.

---

## 2. New contract types (add to `src/contracts.py`)

```python
Tier = Literal["A", "B", "ABSTAIN"]
ToolName = Literal["is_a", "disjoint", "anatomy_of", "compose_laterality",
                   "get_exclusion_list", "retrieve"]   # 6 symbolic; visual added later

class Action(BaseModel):
    tool: ToolName
    args: Dict[str, Any]           # e.g. {"node": "consolidation", "target": "pulmonary_abnormality"}
    kind: Literal["symbolic", "visual"] = "symbolic"

class Observation(BaseModel):
    result: Any                    # path list, bool, anatomy name, exclusion list...
    ok: bool                       # did the tool return a usable result

class TreeNode(BaseModel):
    state_facts: List[PerceptualFact]   # evidence (may grow when visual tools added)
    history: List[Tuple[Action, Observation]]
    answer: Optional[str] = None        # set when agent emits Answer[...]
    reward: float = 0.0                 # closure-progress from verifier
    parent_id: Optional[int] = None
    reflection: str = ""                # search heuristic only — NOT part of trace

class SearchResult(BaseModel):
    answer: str                    # answer string, or "" on abstain
    tier: Tier
    path: List[Tuple[Action, Observation]]   # winning root→leaf path = trace
    conf: float                    # min fact-confidence used in path
```

`Candidate` (the v2 multi-head type) is **removed** — v4 has one agent, no heads.

---

## 3. Verifier (`src/engine/verifier.py`)

Two roles, one module. Deterministic. No weights.

### 3.1 `closure_progress(node, query, dag) → float` — the search value
Dense signal in `[0, 1]` guiding which node to expand next. By question type:

| type | progress signal |
|---|---|
| existential | `1.0` if a witness `is_a*` path to target exists in history; else partial (target resolvable = 0.2) |
| negation | fraction of `get_exclusion_list(target)` items already checked-and-absent; `0.0` if any present (will FAIL) |
| relational | `1.0` if `anatomy_of`/`compose_laterality` resolved for the target fact; else 0.2 |
| counting | `1.0` once distinct observed concepts counted |

Consistency penalty: if history asserts two facts that are `disjoint`, reward `= 0`
(dead branch — backtrack).

**LLM self-eval is never used here.** This is the whole point.

### 3.2 `verify(node, query, dag) → SearchResult` — the terminal gate
Runs when the agent emits an answer. Checks:
1. Every symbolic action's observation is reproducible on the DAG (consistency).
2. The answer follows deterministically from the last/closing observation (soundness).
3. The closure condition for the type is met (completeness).

Verdict → tier:
- All three hold, type ∈ {existential, negation, relational, counting} → **Tier A**.
- Open / outside ontology, or answer present but closure not provable → **Tier B** (flagged).
- Negation with missing exclusion list, or no closure within budget → contributes to **ABSTAIN**.

Closure conditions (carried over from `proposers/perc.py` per-type logic):
- existential: one witness suffices (monotone, early-exit).
- negation: **every** exclusion-list item must be checked absent; missing list → abstain.
- relational: target fact found + attribute resolved.
- counting: count of distinct observed concepts.

---

## 4. Symbolic tool layer (`src/tools/symbolic.py`)

Thin wrappers turning `OntologyDAG` methods into the uniform `Action → Observation`
interface the search calls. No new logic — `dag.py` already implements all of it.

```python
def run_tool(action: Action, facts, dag, img_wh) -> Observation:
    # dispatch on action.tool:
    #   is_a              -> dag.reachable_is_a(node, target)        -> path or None
    #   disjoint          -> dag.disjoint(a, b)                      -> bool
    #   anatomy_of        -> dag.anatomy_of(bbox, w, h)              -> zone or None
    #   compose_laterality-> dag.compose_laterality(bbox, w, h)      -> "left"/...
    #   get_exclusion_list-> dag.get_exclusion_list(name)            -> list or None
    #   retrieve          -> NotImplemented (later spec)             -> ok=False
```

`run_tool` is pure: same `(action, facts, dag)` → same `Observation`.

---

## 5. Tree search (`src/search/tree_search.py`)

**Best-first** (not full MCTS — branching is small; justify empirically later).

```python
def search(query, facts, dag, agent, budget=20, k=3) -> SearchResult:
    root = TreeNode(state_facts=facts, history=[])
    frontier = [root]                       # nodes not yet expanded
    while frontier and nodes_expanded < budget:
        node = pop_highest_reward(frontier)         # best-first select
        if node.answer is not None:
            result = verify(node, query, dag)
            if result.tier == "A":
                return result                       # PASS → done
            # else fall through: keep searching other branches (backtrack)
        for action in agent.propose_actions(node, query, k):   # expand: k candidates
            child = apply(node, action, facts, dag)            # run_tool -> new node
            child.reward = closure_progress(child, query, dag)
            frontier.append(child)
    return best_tier_b_or_abstain(expanded_nodes, query, dag)
```

- **Select:** highest `reward` (verifier closure-progress).
- **Expand:** `agent.propose_actions` returns up to `k` candidate actions (or an Answer).
- **Backtrack:** implicit — a low-reward branch simply loses to a sibling in the frontier.
- **Reflection:** on a dead branch the agent may attach `reflection` (heuristic only).
- **Terminate:** first Tier-A PASS, or budget exhausted → best Tier-B, else ABSTAIN.

### Agent interface (`src/agent/base.py`)
```python
class Agent(Protocol):
    def propose_actions(self, node: TreeNode, query: Query, k: int) -> list[Action | Answer]:
        ...
```
`MockAgent` (`src/agent/mock.py`): returns scripted actions keyed by question type,
for deterministic tests. LLaVA-Med implements this same Protocol later.

---

## 6. Code style

- Match existing modules: `pydantic` frozen models, `networkx`, plain functions.
- Flat `if/elif` over abstractions; short single-purpose functions (per CLAUDE.md).
- No learned weights, no randomness anywhere in this spec's modules.
- Reuse `dag.py` and `proposers/perc.py` logic — do not reimplement reachability/IoU.

---

## 7. Testing strategy

Deterministic unit tests, no GPU, using `MockAgent` + the running example from
`docs/project.tex`.

`tests/test_verifier.py`:
- existential: witness present → progress 1.0, verify → Tier A.
- negation: all exclusions absent → Tier A "absent"; one present → FAIL; missing list → ABSTAIN.
- relational: anatomy resolved → Tier A with zone.
- counting: 3 facts → answer "3".
- disjoint violation → reward 0.

`tests/test_tree_search.py`:
- Existential 2-hop (`consolidation → ... → pulmonary_abnormality`): search finds witness, returns Tier A, path == `reachable_is_a` path.
- Backtrack: mock agent proposes a dead branch first, valid branch second → search still returns Tier A.
- Budget exhausted, nothing verifies → ABSTAIN.
- **Determinism:** same `(query, facts)` → identical `SearchResult` across 100 runs.
- **Deletion test (mock):** remove the witness fact → answer flips (proves path is load-bearing).

---

## 8. Boundaries

### Always
- Reward = verifier closure-progress (deterministic). Path = the actual search trace.
- Abstain when ontology coverage is insufficient (esp. negation w/ missing list).
- Keep search, verifier, tools deterministic and GPU-free.

### Ask first
- Visual-tool verifier semantics (how `re_detect` results fold into the graph) — later spec.
- Whether best-first needs upgrading to MCTS (only if benchmarks show branching too large).
- EvaX swap (interface is fixed; timing is open).

### Never
- Use LLM self-evaluation as the search reward or the answer selector.
- Use learned weights, sampling, or non-determinism in core modules.
- Mutate the DAG or the base evidence at runtime.
- Let a visual action replace base EvaX evidence (would break closed-world negation).
```
