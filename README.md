# KRONOS

**Knowledge-Routed Neuro-Symbolic Reasoning for Faithful Chest X-ray VQA**

KRONOS answers chest X-ray questions with an auditable trace grounded in a
knowledge graph. A frozen VLM (MedGemma-4b) proposes actions; a deterministic
verifier on a clinical ontology decides the answer. The verifier — never the
model — is the authority.

One unified tree search engine handles all question types — from single-image
VQA ("Is there Cardiomegaly?") to multi-hop shared-cause reasoning ("Can one
condition cause both A and B?"). The verifier on the KG guides search and
decides the answer; the LLM only proposes which tools to call or edges to
explore. No model is fine-tuned.

---

## How it works

```
Chest X-ray ─► YOLOv12 detector (frozen) ─► PerceptualFacts (finding, bbox, conf)
Question    ─► rule-based parser          ─► Query(type, target)
                                                │
              ┌─────────────────────────────────┘
              ▼
  Unified best-first tree search (all question types)
  MedGemma proposes tool calls / graph edges to explore
  Verifier scores each node (closure_progress) + decides tier
  Tools:
    symbolic : is_a, disjoint, anatomy_of, compose_laterality,
               get_exclusion_list, neighbors, find_path
    visual   : inspect, re_detect, compare
    retrieval: retrieve (BiomedCLIP + FAISS)
              │
              ▼
  SearchResult(answer, tier={A,B,ABSTAIN}, trace)
  Tier A = every step verified on KG (existential/negation/shared_cause/...)
  Tier B = perception-only (open questions)
  ABSTAIN = insufficient evidence
```

## Repository structure

```
src/
  contracts.py              # Pydantic types: Action, Observation, TreeNode, SearchResult
  pipeline.py               # End-to-end run() for single-image VQA
  data/loaders.py           # Dataset loaders (VinDr-VQA, ChestAgentBench, multi-hop)
  agent/
    medgemma.py             # MedGemma-4b agent (frozen, 4-bit quantized)
    mock.py                 # MockAgent for deterministic tests (no GPU)
    prompt.py               # Prompt builder + output parser
  engine/verifier.py        # closure_progress (search value) + verify (tier gate)
  search/tree_search.py     # Best-first tree search + reflection backtrack
  ontology/dag.py           # Ontology DAG (is-a, disjoint, anatomy, laterality, causal)
  perception/detector.py    # YOLOv12 wrapper → PerceptualFact list
  question/parser.py        # Rule-based question typing
  retrieval/                # BiomedCLIP encoder + FAISS RAG index
  eval/
    predictors.py           # Multi-hop predictors: mock, zero_shot, cot, react, kronos (ToG)
    multihop_metrics.py     # Grading: grounding_rate, hallucination, load-bearing
  tools/
    symbolic.py             # 6 symbolic tools (wrap ontology DAG)
    visual.py               # 3 visual tools (inspect, re_detect, compare)
    dispatch.py             # Unified router: symbolic vs visual

scripts/
  build_kg.py               # Map VinDr findings → RGO, extract causal subgraph
  build_multihop_qa.py      # Generate multi-hop shared-cause QA items
  run_multihop.py           # Run a predictor over the QA set → predictions JSONL
  eval_multihop.py          # Grade predictions → metrics table

data/
  ontology/                 # dag.yaml, causal_kg.yaml, exclusion_lists.yaml, anatomy_zones.yaml
  vindr_cxr_vqa/            # VinDr-CXR images + VQA annotations (train.csv, vqa.json)
  chestagentbench/          # ChestAgentBench metadata + figures
  multihop_qa/              # Generated shared-cause QA (qa.jsonl)
  rag/                      # BiomedCLIP FAISS index + case JSONL

tests/                      # 375 deterministic tests (CPU) + 6 GPU integration tests
```

## Quickstart

### Tests (CPU, no GPU)

```bash
conda activate medcxr
pytest tests/ -m "not gpu"     # 375 tests, ~45s
```

### Multi-hop evaluation

```bash
# Run the ToG predictor (needs GPU for MedGemma)
python scripts/run_multihop.py --system kronos --limit 50

# Grade predictions
python scripts/eval_multihop.py --preds results/preds_kronos.jsonl

# Ablations
python scripts/run_multihop.py --system single_hop    # max_depth=1
python scripts/run_multihop.py --system no_prune      # skip LLM pruning

# Baselines (zero_shot, cot, react need GPU; mock is CPU-only)
python scripts/run_multihop.py --system mock
```

### Single-image VQA

```python
from src.pipeline import run
from src.ontology.dag import OntologyDAG
from src.perception.detector import Detector
from src.agent.medgemma import MedGemmaAgent

dag = OntologyDAG("data/ontology/dag.yaml",
                  "data/ontology/exclusion_lists.yaml",
                  "data/ontology/anatomy_zones.yaml")
detector = Detector("weights/yolov12s_vindr.pt", dag=dag)
agent = MedGemmaAgent(quantize=True)

result = run("image.png", "Is there Cardiomegaly?", dag, detector, agent)
print(result.answer, result.tier, result.path)
```

## Key design decisions

- **Verifier-as-value:** the search tree is guided by a deterministic verifier on
  the KG. The LLM never evaluates itself — fixing the reliability problem of
  Tree of Thoughts.
- **Witness binding:** an `is_a` only counts as evidence when its source is a
  *detected* fact, preventing ontology tautologies from producing false answers.
- **Think-on-Graph for multi-hop:** the LLM explores the causal KG by pruning
  edges; the verifier (not the LLM) decides when a path is sufficient.
  Every trace edge is a real KG edge by construction.
- **Faithfulness metrics:** grounding rate (every edge is real + chain connects
  both findings), load-bearing rate (removing the chain flips the answer),
  hallucination rate (named cause not in the gold set).

## Environment

- Python 3.10+, `medcxr` conda env
- GPU: ~3 GB VRAM (MedGemma 4-bit) + ~0.5 GB (BiomedCLIP + YOLOv12)
- Spec: [docs/KRONOS_reasoning_SPEC.md](docs/KRONOS_reasoning_SPEC.md)
