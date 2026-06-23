# Graph-Reasoning KRONOS — Specification

**Status:** draft for approval
**Target:** near-term workshop / arXiv (~6–8 week MVP)
**Date:** 2026-06-23

---

## 1. Objective

**One-sentence contribution:** A frozen multimodal model proposes *graph operations* over a
radiology knowledge graph; best-first tree search explores them; a deterministic verifier
validates them; and the **validated multi-hop graph path is the answer's explanation** — reasoning
*through the graph*, not prose chain-of-thought.

**Headline claim (faithfulness-first):** every Tier-A answer carries a graph path a human can
check, and the system hallucinates less than CoT/ReAct baselines — while staying *competitive*
(not necessarily SOTA) on answer accuracy.

**Hard constraints**
- The model (MedGemma 4B) is **frozen**. No fine-tuning, no LoRA, no training of any kind.
  Improvement comes only from in-context prompting + graph + search + verifier.
- The contribution is **multi-hop graph traversal** (PoG-style), not single-hop lookups and not
  prose CoT.
- All headline metrics are **automatic and deterministic** — no human judge, no LLM judge.

**Grounding in prior work**
- *MedReason* — KG "thinking paths" as faithful rationale. We use such paths at **inference time**,
  frozen, instead of distilling them into weights.
- *Plan-on-Graph (PoG)* — model-guided KG traversal with self-correction. We adopt the
  traversal + reflection loop.
- *Tree-of-Thought / LATS* — search over reasoning states with a value function. Our value
  function is the deterministic verifier (`closure_progress`).

**Honesty note for the paper:** RGO relations are *"may cause"* (differential-diagnosis
plausibility), so a rationale is an **evidence-consistent differential path**, not a proof of
causation. State this explicitly; do not overclaim.

---

## 2. What already exists (reuse, do not rebuild)

KRONOS at HEAD `8cfe145` already provides:
- `src/ontology/dag.py` — ontology graph + `is_a`, `disjoint`, `anatomy_of`, `compose_laterality`,
  `get_exclusion_list`, `resolve_slug`, `reachable_is_a`.
- `src/engine/verifier.py` — `closure_progress` (search value) + `verify` (Tier A / B / ABSTAIN).
  Tier B = "model answered, path not verified" — the natural hook for reflection.
- `src/search/tree_search.py` — best-first search; `TreeNode.reflection` field exists but is
  **never written** (dead feedback channel to wire).
- `src/agent/prompt.py`, `src/agent/medgemma.py` — frozen proposer; prompt already reads
  `node.reflection`.
- `src/tools/{symbolic,visual,dispatch}.py` — Action → Observation tool layer.
- `scripts/eval_vindr_vqa.py` — VinDr-CXR-VQA harness.
- `tests/` — 29+ green tests, incl. `test_deletion_flips_answer` (the seed of the load-bearing
  metric).

This spec **extends** this system. It does not replace the verifier-first design.

---

## 3. What to build (MVP scope)

### 3.1 Knowledge graph layer
Source files are already in `data/`: `RGO-2.0.owl` (`may_cause` graph, namespace
`gamuts.net/RGO/`), `RadLex.owl` + `PunRadLex4.3.owl` (anatomy). Parse with **stdlib** (no
`rdflib` dependency).

**Mapping the 14 VinDr findings → RGO is the hard part and the main risk** — it gates KG quality
and therefore the whole faithfulness claim. Inspection (2026-06-23) showed RGO concepts carry
`hasDbXref RADLEX:RIDxxxx`, but only **4/14** findings bridge by RID (pleural effusion,
calcification, atelectasis, pneumothorax). So use **multi-key mapping**, in order:
1. **RID xref** — finding's RID (from `synonyms.yaml`) vs RGO `hasDbXref RADLEX:`.
2. **Exact label** — finding name vs RGO `rdfs:label` (recovers e.g. cardiomegaly, lung
   opacity→"pulmonary opacity", pulmonary fibrosis, ILD→"interstitial lung fibrosis").
3. **Synonym** — `synonyms.yaml` curated synonyms vs RGO labels.
4. **(optional) SNOMED bridge** — RGO has many `hasDbXref SNOMEDCT:`; use if RadLex supplies a
   finding's SNOMED id.
5. **Manual** — for the residual, the assistant emits candidate RGO concepts for a human to pick.

**Honesty constraints on mapping**
- Mapping only recovers concepts that *exist* in RGO under another name; it must **not invent**
  a concept. Findings that are pure imaging descriptors with no RGO disorder/observation (e.g.
  "Aortic enlargement") are either mapped by hand to the nearest clinical concept (clinician
  sign-off) or kept as **observation-only** nodes that join the graph via RadLex **anatomy**
  rather than `may_cause`.
- The ~10 non-trivial mappings need **clinical verification**.

**Build is two phases (see `scripts/build_kg.py`)**
- *Phase 1 — mapping assistant:* parse RGO, auto-map via keys 1–3, and for anything unmapped emit
  candidate concepts. Output `data/ontology/vindr_to_rgo.yaml` (auto rows + `MANUAL` rows) +
  a coverage report. Human verifies/fills the `MANUAL` rows.
- *Phase 2 — subgraph extraction (after mapping verified):* from the mapped seeds, extract the
  N-hop `may_cause` (+ RadLex anatomy) subgraph; emit a static artifact loaded like the existing
  DAG.

- **License check** before redistribution (RGO OWL + RadLex are free; confirm terms).

### 3.2 Multi-hop graph operations (new tools)
Add two operations, named to match the existing tool style:
- `neighbors(node, relation, direction)` — return adjacent concepts along a relation
  (e.g. what a disorder *may_cause*; what *may_cause* an observation). Lets the model **choose
  what to walk** (PoG-style frontier expansion).
- `find_path(source, target, relations)` — return a multi-hop path connecting source to target
  across allowed relations, or `None`. This is the **multi-hop witness** the verifier checks.

The model proposes `neighbors`/`find_path`/existing ops as actions; search builds the walked
subgraph; the **path returned by `find_path` is the rationale**.

### 3.3 Wire the reflection loop (verifier as critic)
- Add `explain(node, query, dag)` to the verifier: a short, deterministic reason a node failed
  to reach Tier A (mirrors the verify branches).
- In `tree_search.py`: when a model answer verifies to **Tier B or ABSTAIN**, re-queue the parent
  state **once** with `reflection = explain(...)` (guard with `not node.reflection` to prevent
  loops). MockAgent ignores reflection → existing tests stay green; MedGemma reads it and retries
  toward Tier A.

### 3.4 Multi-hop evaluation subset
- **Construct** a QA subset whose questions *require* a multi-hop chain (e.g. "Is there a finding
  that may cause X?", "Which detected finding is associated with Y?"), generated from VinDr bbox
  labels + the KG, each tagged with required hop-count.
- *Design details (templates, hop tagging, balancing) are a TBD to be specified before
  implementation — see §8.*

### 3.5 Baselines
1. MedGemma zero-shot.
2. MedGemma + CoT (the "prose reasoning" contrast).
3. MedGemma + ReAct over the **same graph tools, no verifier** (isolates verifier/search value).
4. KRONOS ablations: single-hop-only (isolates multi-hop value); no-reflection (isolates loop).

### 3.6 Automatic faithfulness metrics
- **Grounding rate** = Tier-A answers backed by a valid graph path / all Tier-A answers.
- **Load-bearing (deletion) test** = delete an edge on the witness path → answer must flip or
  abstain (generalize the existing `test_deletion_flips_answer`).
- **Hallucination rate** = answer asserts a finding outside the verified fact set (rule over the
  closed 14-finding vocab — reproducible).
- **Tier discipline** = ABSTAIN/Tier-B handled correctly (no overclaiming).
- Secondary: VinDr-CXR-VQA accuracy must **not regress**.

---

## 4. Commands

```
# Build the CXR-scoped multi-hop KG from RGO + RadLex sources
python scripts/build_kg.py --out data/ontology/

# Run full graph-reasoning pipeline on the multi-hop subset
python scripts/eval_multihop.py --system kronos

# Run a baseline (zero_shot | cot | react | ablation_singlehop | ablation_noreflect)
python scripts/eval_multihop.py --system <baseline>

# Standard VinDr-CXR-VQA (regression check)
python scripts/eval_vindr_vqa.py

# Tests
conda run -n medcxr python -m pytest -q
```

---

## 5. Project structure (additions in **bold**)

```
src/
  ontology/dag.py            # extend: load multi-hop relations, neighbors(), find_path()
  engine/verifier.py         # extend: explain(); multi-hop witness in _verify_existential
  search/tree_search.py      # extend: wire reflection re-queue
  tools/symbolic.py          # extend: neighbors, find_path dispatch
  agent/prompt.py            # extend: describe new ops; inject KG neighborhood as grounding
scripts/
  build_kg.py                # NEW: RGO + RadLex -> compact CXR KG artifact
  eval_multihop.py           # NEW: run kronos + baselines, emit metrics
data/ontology/
  vindr_to_rgo.yaml          # NEW: 14 findings -> RGO/RadLex concept ids (reviewable mapping)
  kg.yaml / kg.graphml       # NEW: built multi-hop graph artifact
tests/
  test_kg_multihop.py        # NEW: neighbors/find_path correctness
  test_explain.py            # NEW: explain() reason per qtype
  test_faithfulness_metrics.py  # NEW: grounding rate + deletion test
```

---

## 6. Code style (match existing KRONOS)

- Flat `if`/`elif` over the query/relation type — no metaprogramming, no clever abstractions.
- Plain params with simple defaults; no heavy typing unless it aids clarity.
- New tools mirror the existing `Action → Observation` wrappers in `symbolic.py`.
- **Deterministic**: same inputs → same outputs (a determinism test already exists; keep it green).
- Top-to-bottom reading order; short single-purpose functions; comment only the non-obvious *why*.
- Surgical: every changed line traces to this spec. Do not refactor adjacent code.

---

## 7. Testing strategy

- **Unit:** `neighbors`/`find_path` on a fixture KG (known multi-hop paths, dead ends, cycles
  guarded); `explain()` returns a sensible reason per qtype.
- **Faithfulness (automatic):** grounding rate computed from search output; deletion test flips or
  abstains on every load-bearing edge; hallucination rule over the 14-finding vocab.
- **Regression:** all current tests stay green; VinDr-CXR-VQA accuracy not worse than HEAD.
- **Determinism:** repeated runs are identical (extend existing determinism test to multi-hop).
- Each build step has a verify check before moving on (goal-driven).

---

## 8. Open TBDs (to specify before/at implementation)

1. **Multi-hop QA subset design** — question templates, hop-count tagging, answer derivation from
   VinDr labels + KG, class balance, size. *(User: "I'll build it, you'll tell me how.")*
2. **RGO/RadLex export specifics** — exact OWL/RDF parse, relation whitelist, hop-limit N for the
   thoracic subgraph, and the VinDr→concept mapping table.
3. **Reflection content** — exact wording of `explain()` per qtype (kept short, actionable).

---

## 9. Boundaries

**Always**
- Keep the model frozen and the verifier/metrics deterministic and reproducible.
- Ground every Tier-A rationale in an actual KG path; report the path.
- Keep changes surgical and the test suite green.

**Ask first**
- Adding any new dependency, dataset, or external service.
- Expanding scope beyond the MVP (human eval, extra datasets, extra baselines, PoG/LATS
  reimplementation) — these are deferred.
- Any change to the KG sources or licensing assumptions.

**Never**
- Fine-tune, LoRA, or otherwise train the model.
- Let the model's free text override a verifier decision (no rubber-stamping unverified answers
  as Tier A).
- Fabricate findings or graph edges; the KG must come from the cited ontologies.
- Introduce non-determinism into the headline metrics (no LLM-judge in the primary results).

---

## 10. Definition of done (MVP)

- [ ] Compact CXR multi-hop KG built from RGO + RadLex, with a reviewable VinDr→concept mapping.
- [ ] `neighbors` + `find_path` ops implemented, tested, dispatched.
- [ ] Reflection loop wired (Tier B/ABSTAIN → one reflective retry); existing tests green.
- [ ] Multi-hop QA subset constructed (design per §8.1).
- [ ] Baselines run: zero-shot, CoT, ReAct-same-tools, two ablations.
- [ ] Automatic metrics reported: grounding rate, deletion test, hallucination rate, tier
      discipline; VinDr-CXR-VQA non-regression.
- [ ] Result tables support the claim: **multi-hop graph reasoning is more faithful than CoT**,
      with competitive accuracy.
```
