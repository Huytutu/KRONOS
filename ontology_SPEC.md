# SPEC — Ontology DAG (`src/ontology/dag.py`)

> References: [README §5.1-§5.3](docs/README.md), [IMPLEMENTATION §5.1](docs/IMPLEMENTATION.md)

---

## 1. Objective

Provide the symbolic engine with a small, curated knowledge graph that supports
four reasoning operations: subsumption, disjointness, anatomy mapping, and
laterality composition. The engine calls these as deterministic graph operations
— no learning, no approximation.

Used by:
- **Engine** — all four question types use DAG queries.
- **Verifier** — same code as engine, in check mode.

**Non-goals:** large-scale ontology (RadLex has ~68k nodes, we need ~40–50).
No learned embeddings. No runtime graph modification.

---

## 2. Data files

### 2.1 `data/ontology/dag.yaml`

Hand-curated, version-controlled. Contains all nodes and edges.

```yaml
# Nodes: each has an id (lowercase slug), display name, and optional RID.
nodes:
  - id: abnormality
    name: Abnormality

  - id: cardiac_abnormality
    name: Cardiac abnormality

  - id: cardiomegaly
    name: Cardiomegaly
    rid: RID1392

  - id: left_lung
    name: Left lung
    type: anatomy

  # ... ~40-50 nodes total

# Edges: relation + source + target.
edges:
  # is-a: child is-a parent (subsumption)
  - relation: is-a
    source: cardiomegaly
    target: cardiac_abnormality

  - relation: is-a
    source: cardiac_abnormality
    target: abnormality

  # part-of: anatomy containment
  - relation: part-of
    source: left_lung
    target: lung

  # disjoint-with: mutual exclusion (symmetric)
  - relation: disjoint-with
    source: pneumothorax
    target: pleural_effusion

  # laterality: which anatomy zone a finding typically appears in
  - relation: laterality
    source: cardiomegaly
    target: midline
```

**Node categories:**
- **Finding nodes** (14 VinDr leaves) — the concepts the system reasons about.
- **Abstraction nodes** (~2 layers) — group findings for subsumption queries.
  Examples: `cardiac_abnormality`, `pulmonary_abnormality`, `pleural_abnormality`.
- **Anatomy nodes** — `left_lung`, `right_lung`, `lung`, `mediastinum`,
  `pleural_space`, `chest`. For "Where is" and laterality composition.
- **Root** — single `abnormality` root that all finding subtrees reach.

**Relations:**
| Relation | Meaning | Symmetric? |
|----------|---------|------------|
| `is-a` | child is a subtype of parent | no |
| `part-of` | child is anatomically contained in parent | no |
| `disjoint-with` | two findings cannot co-occur in the same region | yes |
| `laterality` | finding is associated with an anatomy zone | no |

### 2.2 `data/ontology/exclusion_lists.yaml`

Per-finding closed-world lists. The engine must check every item on this list
before concluding absence (negation questions). If the list is missing or
incomplete for a finding, the engine returns `UNVERIFIABLE`.

```yaml
# For negation: "Is there no X?" — engine checks every finding in E(X).
# If none are observed, answer = absent. If list is missing, abstain.

Cardiomegaly:
  - cardiomegaly
  - cardiac_hypertrophy

Pneumothorax:
  - pneumothorax

Pleural effusion:
  - pleural_effusion
  - hydrothorax

# ... one entry per finding that negation questions can target
```

**This file needs clinical review.** The lists determine soundness of negation
answers. Start with a minimal list (each finding excludes only itself) and
expand with clinician input.

---

## 3. Module interface

File: `src/ontology/dag.py`

Method names follow IMPLEMENTATION.md §5.1 (authoritative). Each maps to one of
the four reasoning roles the engine uses (README §5.2 / project.tex §Bước 4).

```python
class OntologyDAG:
    def __init__(self, dag_path, exclusion_path=None):
        # Load dag.yaml into a networkx DiGraph.
        # Optionally load exclusion_lists.yaml.

    # --- Role 1: subsumption (is-a) ---
    def reachable_is_a(self, node, target):
        # Return the is-a path [node, ..., target] if node is-a* target,
        # else None. The path IS the derivation the engine cites.
        # reachable_is_a("cardiomegaly", "cardiac_abnormality")
        #   -> ["cardiomegaly", "cardiac_abnormality"]

    def is_a(self, node, target):
        # bool convenience wrapper: reachable_is_a(...) is not None.

    # --- Role 2: disjointness ---
    def disjoint(self, a, b):
        # True if a and b are declared disjoint-with (symmetric).

    # --- Role 3: anatomy mapping (bbox -> anatomy node via IoU) ---
    def anatomy_of(self, bbox):
        # Return the anatomy node whose zone has highest IoU with bbox,
        # or None if no zone overlaps above threshold. Answers "Where".

    # --- Role 4: laterality composition ---
    def compose_laterality(self, finding, bbox):
        # Return "left" | "right" | "bilateral" | "midline" for a finding
        # at bbox (from bbox position relative to image midline).

    # --- closed-world negation support ---
    def get_exclusion_list(self, finding_id):
        # Return the closed-world list for a finding, or None if no list
        # exists (engine then returns UNVERIFIABLE -> abstain).

    # --- helpers ---
    def get_node(self, node_id):              # node dict or None
    def get_node_by_name(self, name):         # linker canonical name -> node_id
    def children(self, node_id):              # direct is-a children
```

Dependencies: `networkx`, `pyyaml`. No dependency on parser, linker, or engine.

**Note on roles 3-4 (spatial).** `anatomy_of(bbox)` and `compose_laterality`
need anatomy *zone coordinates* (normalized boxes for lung/mediastinum/pleural
space). These are a separate data artifact (`anatomy_zones.yaml`) and depend on
the perception bbox convention being fixed. Build roles 1-2 first; roles 3-4
follow once zones are defined (see plan Phase B).

---

## 4. Node inventory (target ~40-50 nodes)

### Finding nodes (14 VinDr leaves)
| Finding | ID | is-a parent |
|---------|-----|-------------|
| Aortic enlargement | `aortic_enlargement` | `vascular_abnormality` |
| Atelectasis | `atelectasis` | `pulmonary_abnormality` |
| Calcification | `calcification` | `abnormality` |
| Cardiomegaly | `cardiomegaly` | `cardiac_abnormality` |
| Consolidation | `consolidation` | `pulmonary_abnormality` |
| ILD | `ild` | `pulmonary_abnormality` |
| Infiltration | `infiltration` | `pulmonary_abnormality` |
| Lung Opacity | `lung_opacity` | `pulmonary_abnormality` |
| Nodule/Mass | `nodule_mass` | `pulmonary_abnormality` |
| Other lesion | `other_lesion` | `abnormality` |
| Pleural effusion | `pleural_effusion` | `pleural_abnormality` |
| Pleural thickening | `pleural_thickening` | `pleural_abnormality` |
| Pneumothorax | `pneumothorax` | `pleural_abnormality` |
| Pulmonary fibrosis | `pulmonary_fibrosis` | `pulmonary_abnormality` |

### Abstraction nodes (~2 layers)
- `abnormality` (root)
- `cardiac_abnormality` is-a `abnormality`
- `pulmonary_abnormality` is-a `abnormality`
- `pleural_abnormality` is-a `abnormality`
- `vascular_abnormality` is-a `abnormality`

### Anatomy nodes
- `chest` (root anatomy)
- `lung` part-of `chest`
- `left_lung` part-of `lung`
- `right_lung` part-of `lung`
- `mediastinum` part-of `chest`
- `pleural_space` part-of `chest`
- `heart` part-of `mediastinum`

**Note:** This inventory is provisional. Some findings may need additional
intermediate nodes or different groupings. Clinician input needed for:
- Whether `calcification` belongs under a more specific parent
- Whether `infiltration` and `consolidation` should share a parent
- Which disjoint-with pairs are clinically correct

---

## 5. Connection: linker canonical name -> DAG node ID

The linker outputs canonical names (e.g., `"Cardiomegaly"`). The DAG uses
lowercase slugs as node IDs (e.g., `cardiomegaly`). Each DAG node stores its
display `name` which matches the linker's canonical name.

Lookup: `dag.get_node_by_name(canonical_name)` — matches `node.name` against
the linker output. This is a simple dict lookup, same pattern as the linker.

---

## 6. Testing

### Unit tests (`tests/test_dag.py`)
- Load dag.yaml successfully
- `is_a("cardiomegaly", "cardiac_abnormality")` -> True
- `is_a("cardiomegaly", "abnormality")` -> True (transitive)
- `is_a("cardiomegaly", "pulmonary_abnormality")` -> False
- `disjoint("pneumothorax", "pleural_effusion")` -> True (if declared)
- `disjoint("pneumothorax", "pneumothorax")` -> False
- `get_exclusion_list` returns list or None
- Every VinDr finding node exists in the DAG
- `children("pulmonary_abnormality")` returns expected set
- `anatomy_of` returns correct anatomy nodes

### Structural tests
- DAG has no cycles (acyclic check)
- All 14 VinDr findings are leaf nodes reachable to root
- Single root node (`abnormality`)
- Every `is-a` edge points from child to parent (direction check)
- disjoint-with is symmetric in practice (if a disjoint b, b disjoint a)

---

## 7. Boundaries

### Always do
- Keep dag.yaml and exclusion_lists.yaml human-readable and version-controlled.
- Return None / empty when data is missing — never invent edges.
- Use networkx for graph operations (standard, well-tested).
- Keep node IDs as lowercase slugs, display names match linker canonicals.

### Never do
- Auto-generate edges from RadLex/SNOMED (defeats "curated" purpose).
- Add nodes not needed by the 14 VinDr findings + their abstraction layers.
- Make the graph mutable at runtime.
- Depend on parser, linker, engine, or perception.

### Ask first
- Which disjoint-with pairs are clinically valid.
- Whether exclusion lists are complete (affects negation soundness).
- Any new abstraction layers beyond the provisional ones in §4.
