# KRONOS

**Knowledge-Routed Neuro-Symbolic Multimodal Reasoning for Faithful Chest X-ray VQA**

KRONOS answers questions about chest X-rays **and** emits a trace that is the *actual cause*
of the answer — not a post-hoc rationalization. A small VLM (LLaVA-Med) searches a **tree** of
visual + symbolic actions, guided by a **deterministic verifier**; the winning root→leaf path
is an auditable, causally-verifiable evidence trace.

> Design v4: *verifier-guided multimodal tree search*. Target venue: MICCAI.
> Full design: [docs/project.tex](docs/project.tex). Specs: [v4_core_SPEC.md](v4_core_SPEC.md),
> [v4_neural_SPEC.md](v4_neural_SPEC.md), [ontology_SPEC.md](ontology_SPEC.md).

---

## How it works

```
Chest X-ray ──► EvaX/YOLO detector (frozen)  ──►  findings + bbox + conf   (query-independent)
Question    ──► concept linking + ontology DAG ──► evidence graph
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
  LLaVA-Med agent searches a TREE of actions:
    • visual : inspect · re_detect · compare      (look back at the image)
    • symbolic: is_a · disjoint · anatomy_of · compose_laterality
                · get_exclusion_list · retrieve   (query the graph)
  guided by the VERIFIER (reward = closure-progress, NOT LLM self-eval)
        │
        ▼
  Verify-gate + tiering:
    Tier A — path verified on the graph  → faithful trace (EF@k + deletion test)
    Tier B — open question               → answer flagged "perception-only"
    Abstain — nothing verifies
```

Key ideas:
- **Verifier-as-value:** the tree is guided by a deterministic verifier, so the winning branch
  is faithful *by construction* (not by an LLM judging itself).
- **Reasoning repairs perception:** `re_detect` lets the agent zoom back into a region to catch
  findings the global scan missed — so reasoning lifts accuracy, not just faithfulness.
- **Frozen, no training:** detector + LLaVA-Med are frozen (prompt-only); engine runs on CPU.

## Repository structure

```
src/
  contracts.py         # pydantic types: Action, Observation, TreeNode, SearchResult, Tier
  pipeline.py          # end-to-end run()
  agent/
    base.py            # Agent Protocol
    mock.py            # MockAgent — scripted, deterministic tests (no GPU)
    llavamed.py        # LLaVA-Med 1.5 agent (prompt-only, fp16/4-bit)
  engine/verifier.py   # closure_progress (reward) + verify (gate)
  tools/
    symbolic.py        # 6 symbolic tools (wrap ontology DAG)
    visual.py          # 3 visual tools (inspect / re_detect / compare)
    dispatch.py        # unified router: symbolic vs visual
  search/tree_search.py# best-first tree search + backtrack
  ontology/dag.py      # curated ontology DAG (is-a, disjoint, anatomy, laterality)
  perception/detector.py  # YOLO detector
  question/parser.py   # question typing
  linking/linker.py    # finding → ontology node
data/ontology/         # dag.yaml, exclusion_lists.yaml, anatomy_zones.yaml, synonyms.yaml
tests/                 # 261 deterministic tests (CPU) + GPU integration tests (skipped)
scripts/               # reasoning_need_diagnostic.py, build_synonyms.py
```

## Running

### Tests (CPU, no GPU)
```bash
pytest tests/                      # 261 deterministic tests
python scripts/reasoning_need_diagnostic.py   # dataset reasoning-need analysis
```

### Full pipeline (GPU server)
```bash
export LLAVAMED_PATH=path/to/llava-med-v1.5-mistral-7b
export YOLO_WEIGHTS=path/to/yolov12s_vindr.pt
export TEST_IMAGE=path/to/sample_cxr.png

pytest tests/test_integration_gpu.py -m gpu -v
```

```python
from src.pipeline import run
from src.ontology.dag import OntologyDAG
from src.perception.detector import Detector
from src.agent.llavamed import LLaVAMedAgent

dag = OntologyDAG("data/ontology/dag.yaml",
                  "data/ontology/exclusion_lists.yaml",
                  "data/ontology/anatomy_zones.yaml")
detector = Detector(YOLO_WEIGHTS, dag=dag)
agent = LLaVAMedAgent(model_path=LLAVAMED_PATH, quantize=True)

result = run("cxr.png", "Is there Cardiomegaly?", dag, detector, agent)
print(result.answer, result.tier, result.path)
```

## Status

- ✅ Deterministic core: contracts, symbolic tools, verifier, tree search, MockAgent.
- ✅ Neural layer: LLaVA-Med agent, visual tools, unified dispatch, pipeline.
- ⏳ Next: run GPU integration on server; EF@k + deletion-test evaluation on VinDr-CXR-VQA.

## Faithfulness

- **Tier A** (existential, negation, relational, counting): the answer is the deterministic
  output of the path; symbolic steps are *sound*, visual steps are *evidence-traceable*.
- **Deletion test:** removing a fact/edge in the path flips the answer → proves the trace is
  causal, not decorative.
- Honest scope: KRONOS does **not** claim provably-sound for neural (visual) steps — only
  evidence-traceable + causal. Open questions are served as **Tier B** and flagged.
