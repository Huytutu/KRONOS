# KRONOS — Implementation Plan

Engineering plan to build the system described in [README.md](README.md) (design) and
[proposal.tex](proposal.tex) (proposal). This document is the bridge from design to code: module
layout, typed interfaces, the diagnostics-first ordering, and milestones mapped to the 12-week roadmap.

> **Guiding rule (from the design):** the only *trained* component is a small optional router. Everything
> else is a frozen model or a deterministic graph procedure. Build the deterministic core first, on
> **oracle perception**, then swap in a real detector. Measure before committing (README §10–11).

---

## 0. Scope of phase 1 (VinDr pilot)

**In:** `Is_there` / `Yes_No` (existential), negation/absence, `Where` (relational/anatomy) on
VinDr-CXR-VQA. Engine, three proposers, arbitration, EF@k + deletion test, diagnostics.

**Deferred:** `How_many` (counting — not in the 3-type closure taxonomy; phase-1 policy = **abstain**),
router training (optional, efficiency only), PadChest-GR / GEMeX scale-up.

**Non-goals:** fine-tuning any vision/LLM backbone; a large ontology; autonomous (no-human) deployment.

---

## 1. What the data actually gives us

Grounding every module in the real files under `data/raw/vindr_cxr_vqa/`:

| File | Schema | Used by |
|---|---|---|
| `train.csv` | `image_id, class_name, class_id(0–14), rad_id, x_min,y_min,x_max,y_max, width,height` | **GT detections** → oracle perception + detector training |
| `vqa.json` | per image: list of `{question, answer, reason, type, difficulty, gt_finding, gt_location}` | the VQA task; `type`/`gt_finding`/`gt_location` are **gold labels** |
| `test.csv` | `image_id, width, height` | inference set |
| `sample_submission.csv` | `image_id, PredictionString = "class conf x1 y1 x2 y2 …"` | submission format |

Two facts that shape the whole plan:

1. **`train.csv` has per-finding GT boxes.** → we can run the engine on **oracle perception** (GT boxes
   as `F_img`) before any detector exists. This isolates the reasoning core from perception error and
   lets us validate EF@k cleanly (faithfulness ≠ correctness, README §6).
2. **`vqa.json` already carries gold `type`, `gt_finding`, `gt_location`.** → the question parser's
   type classification and the concept linker are **directly measurable** against gold; EF@k uses
   `gt_location` (`<loc_x1_y1_x2_y2>` token) as the gold region. `reason` is GPT-synthetic → **score EF
   on region only**, never on the reasoning text.

VinDr-CXR finding set (class_id): `0 Aortic enlargement, 1 Atelectasis, 2 Calcification, 3 Cardiomegaly,
4 Consolidation, 5 ILD, 6 Infiltration, 7 Lung Opacity, 8 Nodule/Mass, 9 Other lesion, 10 Pleural
effusion, 11 Pleural thickening, 12 Pneumothorax, 13 Pulmonary fibrosis, 14 No finding`. **14 findings →
DAG leaf set is closed and tiny.**

---

## 2. Repo layout

```
kr-nesy-rag/
├── data/
│   ├── raw/vindr_cxr_vqa/            # given
│   └── processed/                    # built: facts.parquet, vqa_typed.parquet, splits/
├── ontology/
│   ├── dag.yaml                      # nodes + is-a/part-of/disjoint/laterality edges (version-controlled)
│   ├── exclusion_lists.yaml          # per-finding closed-world lists E(c)
│   └── synonyms.yaml                 # finding-name → canonical node
├── kronos/
│   ├── types.py                      # Fact, Query, Candidate, Verdict, Derivation (dataclasses)
│   ├── data/loaders.py               # vqa.json + train.csv → typed records; loc-token parse
│   ├── perception/
│   │   ├── oracle.py                 # GT boxes → F_img   (phase-1 default)
│   │   └── detector.py               # YOLO wrapper (frozen)  (phase-3 swap-in)
│   ├── parser/question_parser.py     # NL → Query; type classifier (+ conservative default)
│   ├── linking/concept_linker.py     # finding-name → DAG node (exact → fuzzy → llm)
│   ├── ontology/dag.py               # load dag.yaml; reachability, disjoint, IoU-anatomy, laterality
│   ├── engine/engine.py              # Engine(Q,K) → ⟨verdict, π⟩  ← the core
│   ├── proposers/{perc.py,sym.py,rag.py}
│   ├── arbitrate/arbitrator.py       # 4-tier + E_perc⊥E_rag conflict → abstain
│   ├── router/router.py              # optional MLP (efficiency only)
│   ├── render/renderer.py            # derivation subgraph → prose (template; LLM optional)
│   └── pipeline.py                   # wires the 9 steps
├── eval/
│   ├── ef_at_k.py  deletion_test.py  risk_coverage.py
│   └── diagnostics.py                # complementarity, reasoning-need, type/linking accuracy
├── scripts/{build_ontology.py, run_diagnostics.py, run_eval.py, infer.py}
└── tests/                            # unit tests per module
```

Stack: Python 3.11, `networkx` (DAG), `pydantic`/`dataclasses` (types), `faiss-cpu` (RAG),
`ultralytics` (YOLO, phase 3), `pandas`/`pyarrow`, `pytest`. Frozen LLM via API for parser/linker only.

---

## 3. Core types (`kronos/types.py`)

```python
@dataclass(frozen=True)
class Fact:                       # one grounded perceptual fact  f_i
    concept: str                  # canonical finding name (e.g. "cardiomegaly")
    bbox: tuple[int,int,int,int]  # (x1,y1,x2,y2) in image px
    laterality: Literal["left","right","bilateral","midline"]
    conf: float                   # [0,1]

QType = Literal["existential","negation","relational","counting"]

@dataclass(frozen=True)
class Query:                      # parsed question  Q
    type: QType
    target: str                   # target concept (DAG node)
    constraints: dict             # e.g. {"attr": "laterality"}

@dataclass(frozen=True)
class Candidate:                  # proposer output
    answer: str
    anchor: object                # regions | subgraph | passages
    head_id: Literal["E_perc","E_sym","E_rag"]
    conf: float

Verdict = Literal["PASS","FAIL","UNVERIFIABLE"]

@dataclass(frozen=True)
class Derivation:                 # π — the faithful trace
    nodes: list[str]
    edges: list[tuple[str,str,str]]   # (src, relation, dst)
    cited_facts: list[Fact]
```

`K` (knowledge base) = `(list[Fact], DAG)`. `loc-token` parser: `<loc_x1_y1_x2_y2>` → `bbox`.

---

## 4. Phase 0 — Diagnostics first (gate, README §11)

**Do this before building proposers — it decides whether the architecture is worth it.** All run on
oracle perception (GT boxes), so no detector needed.

1. **Reasoning-need stats** (`diagnostics.py`): for each VQA item, does answering require subsumption
   (`gt_finding` ≠ asked concept, needs an `is-a` hop) vs pure name-match? Tally per `type`. Justifies
   the engine's existence.
2. **Head-complementarity** (dry run): simulate `E_perc` (name-match), `E_sym` (DAG hop), `E_rag`
   (retrieval) on the train split; % of questions where *exactly one* head is correct. If `E_sym`≈`E_perc`
   ~95%, the routing story is weak — report and reconsider before proceeding.
3. **Type distribution**: map `vqa.json` `type` → `QType`; measure counting/negation share to size the
   closed-world effort.

**Exit gate:** complementarity meaningfully > 0 **and** a non-trivial reasoning-need slice. Else stop and
re-scope.

---

## 5. Modules, in build order

### 5.1 Ontology DAG (`ontology/dag.yaml`, `ontology/dag.py`) — week 3–4
- 14 VinDr leaves + ~2 abstraction layers (`cardiac_abnormality`, `pulmonary_abnormality`,
  `pleural_abnormality`, `abnormality`) + anatomy (`left/right lung`, `mediastinum`, `pleural_space`).
  Target ~40–50 nodes.
- Relations: `is-a`, `part-of`, `disjoint-with` (e.g. `pneumothorax ⊥ pleural_effusion` on pleural
  space), `laterality`. Hand-authored, reviewed by one clinician, **version-controlled**.
- `exclusion_lists.yaml`: per finding, the set that must be checked before concluding absence.
- API: `reachable_is_a(node, target) -> path|None`, `disjoint(a,b) -> bool`,
  `anatomy_of(bbox) -> node` (IoU vs anatomy zones), `compose_laterality(finding,bbox) -> str`.
- **Tests:** golden assertions (`cardiomegaly is-a cardiac_abnormality`, `pneumothorax ⊥ effusion`).

### 5.2 Engine (`engine/engine.py`) — week 5–6 — **the core, build before proposers**
```python
def Engine(Q: Query, K: KB) -> tuple[Verdict, Derivation | None]:
    # existential : first witness v in observed with reachable_is_a(v, Q.target) → PASS, early-exit
    # negation    : check EVERY item in exclusion_lists[Q.target]; none observed → PASS(absent)
    #               exclusion list incomplete for target → UNVERIFIABLE (→ abstain)
    # relational  : traverse ≤ budget B hops (anatomy/laterality); else UNVERIFIABLE
    # counting    : (phase 1) → UNVERIFIABLE
```
- Four checks = subsumption / disjointness / laterality / anatomy (deterministic graph ops).
- `verify(candidate, K) = Engine(query_from_candidate(candidate), K)` — **same code, check mode**.
- Determinism test: same `(Q,K)` → identical `⟨verdict,π⟩` byte-for-byte across runs.

### 5.3 Perception — oracle then detector
- `perception/oracle.py` (week 5): `train.csv` rows → `list[Fact]` (GT boxes, conf=1.0, laterality from
  bbox x-centroid vs midline). **Phase-1 default** — decouples reasoning from perception error.
- `perception/detector.py` (week 10+): frozen YOLO trained on `train.csv`; same `list[Fact]` output.
  Confidence threshold τ=0.5. Swappable behind one interface.

### 5.4 Question parser (`parser/question_parser.py`) — week 5
- Frozen LLM, structured (JSON-schema) output → `Query`. **Parse, not decompose** (closed schema, no
  answer slot).
- Type classifier = a field of the parse; **conservative default on ambiguity** (→ negation/relational,
  never early-exit). Validate against gold `type` in `vqa.json`; report `type_accuracy`.

### 5.5 Concept linker (`linking/concept_linker.py`) — week 5
- exact (synonyms.yaml) → fuzzy (embedding) → LLM fallback. On VinDr the leaf set is closed → mostly
  exact lookup. Validate against `gt_finding`; report `linking_accuracy` separately.

### 5.6 Proposers (`proposers/`) — week 6
- `perc.py`: name-match in `F_img`; anchor = boxes.
- `sym.py`: `Engine(Q,K)` in find-answer mode; anchor = `π`. **Operates on linked facts, not pixels.**
- `rag.py`: image embedding → FAISS top-k similar train cases → majority finding. **Independent of
  `F_img`** (the hedge). Anchor = case ids.

### 5.7 Arbitrator (`arbitrate/arbitrator.py`) — week 7
- 4 tiers (README §4). Plus: **`E_perc` ⊥ `E_rag` disagree on a perceptual fact → lean ABSTAIN/flag**
  (two independent perceptual paths conflict ⇒ low trust).

### 5.8 Renderer (`render/renderer.py`) — week 7
- Template: "Detected {finding} ({conf}, box). {finding} is-a {target}. ∴ {answer}." LLM optional, adds
  no inference. Every sentence traces to an edge in `π`.

### 5.9 Router (`router/router.py`) — optional, week 8+
- Small MLP, **faithfulness-supervised** label = head that *passed verification AND was correct*.
  Wrong router ⇒ slower, never wrong. Skippable in phase 1.

---

## 6. Evaluation harness (`eval/`) — week 8–12

| Metric | File | Definition |
|---|---|---|
| **EF@k** | `ef_at_k.py` | IoU/overlap of `π`'s cited regions vs `gt_location`. **Region only** (reason is synthetic). |
| **Deletion test** | `deletion_test.py` | for each `e_i ∈ π`: remove from `K`, re-run engine; rate of answer-flips. High ⇒ causal. |
| **Risk–coverage** | `risk_coverage.py` | selective accuracy as abstention varies — the **primary** lens, not raw top-1. |
| Task acc / F1 | `risk_coverage.py` | comparability with prior work. |
| type / linking acc, coverage %, abstention rate | `diagnostics.py` | reported, not hidden. |

**Headline ablations (acceptance hinges, README §8):** (1) faithfulness- vs accuracy-supervised router;
(2) full system vs **dense-retrieval-flat** baseline (stand this up *early*, week 8); (3) propose-then-
verify vs single best head vs naive fusion, at matched coverage.

---

## 7. Milestones (maps to proposal §Lộ trình 12 tuần)

| Wk | Deliverable | Done when |
|---|---|---|
| 1–2 | **Diagnostics** (phase 0) on oracle perception | complementarity + reasoning-need numbers; exit gate passed |
| 3–4 | `dag.yaml` v1.0 + exclusion lists | golden ontology tests green; clinician sign-off |
| 5–6 | Engine + oracle perception + parser + linker | `Engine(Q,K)` passes determinism + closure unit tests end-to-end |
| 7 | 3 proposers + arbitrator + renderer | full pipeline answers VinDr with traces + tiers |
| 8–9 | dense-retrieval-flat baseline + ablations | comparison table; EF/selective-acc vs baseline |
| 10–12 | EF@k + deletion test + detector swap-in; PadChest-GR negation probe | causal trace evidence; results frozen for write-up |

---

## 8. Testing & reproducibility
- Unit tests per module; **golden-file** tests for the engine (fixed `K` → fixed `π`).
- Determinism CI check on the engine. Frozen backbones pinned by hash. `dag.yaml`/exclusion lists
  version-tracked; track **per-finding closed-world coverage %** as a first-class number.
- Everything in phase 1 runs CPU-only except the optional detector/router.

## 9. Key risks carried from design (README §10)
Low head-complementarity (measured wk1–2, gate), weak deletion test (protocol designed wk10), losing to
flat retriever (baseline wk8), curation not scaling (track coverage %, abstain on tail), concept-linking
correlated error (`E_rag` hedge + separate linking acc). **No A* promise before the four headline
experiments produce numbers — those are the gate.**
