# SPEC — KRONOS `E_rag` (the `retrieve` tool)

> Scope: implement the `retrieve` tool — the one piece of the v4 action set still
> stubbed (`Observation(ok=False)` in [src/tools/symbolic.py:37](src/tools/symbolic.py#L37)).
> `E_rag` is dense **FAISS retrieval over a VinDr-CXR case corpus**, embedded by a
> **frozen CXR encoder**, exposed as a tool the agent can call in the ReAct loop.
> It is the **"second opinion"**: deliberately independent of the EvaX/YOLO
> perceptual facts, so it does not share their correlated errors
> ([project.tex:453](docs/project.tex#L453)).
>
> **Prerequisite:** v4 core + neural layers built and passing (contracts, verifier,
> dispatch, tree search, pipeline).
>
> References: [v4_core_SPEC.md](v4_core_SPEC.md), [v4_neural_SPEC.md](v4_neural_SPEC.md),
> [intent v3](docs/intent/kronos-v3-vlm-agent.md), [project.tex](docs/project.tex)
> (action table line 623, RAG line 883).

---

## 0. Decisions locked with the user

| Decision | Choice |
|---|---|
| Faithfulness role | **Advisory only.** retrieve informs the agent's *next* action and may support Tier-B. It **never** closes a Tier-A path. Tier-A stays pure deterministic graph-ops. |
| Embedding source | **Frozen CXR encoder** (BiomedCLIP), separate from LLaVA-Med and EvaX → true independent second opinion. |
| Corpus | **VinDr-CXR-VQA** (`data/vindr_cxr_vqa/vqa.json`), **train split only** (eval images never indexed → no leakage). |
| Case content | The **`reason`** field(s) — per-finding clinical reasoning — are the "second opinion" text. ✅ confirmed |
| Dependencies | `faiss-cpu` + `open_clip_torch` approved. ✅ confirmed |
| Build scope | **Full**: real encoder + real FAISS index built from the corpus + the tool, plus CPU-testable interface. |

## 0.1 Assumptions — correct me before I build

1. **Case content = `reason`.** `vqa.json` groups QA by `image_id`; each QA carries a
   `reason` (clinical rationale) and a `gt_finding`. A case = `{case_id=image_id,
   labels=distinct gt_finding, report=unique reasons joined}`. The reasons are the advisory
   "second opinion" the agent reads. (Resolved from the earlier label-set assumption.)
2. **Encoder = BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`),
   frozen, image-tower only, 512-d L2-normalised embeddings. MedCLIP is the fallback if
   you prefer. Loaded behind a GPU/optional-import boundary like LLaVA-Med.
3. **FAISS `IndexFlatIP`** (exact inner-product on normalised vectors = cosine). Corpus is
   small enough that exact search is fine and fully deterministic. No IVF/HNSW.
4. **The query image is embedded once per run** and cached on the retriever
   (`set_image`, mirroring the agent). The agent calls `retrieve` with just `{"k": n}`.
5. Retrieval is **available-or-graceful**: with no index/encoder/query-embedding,
   `retrieve` returns `ok=False` (dead branch → backtrack), exactly like the visual tools
   degrade when `image is None`.

---

## 1. Objective

Turn `retrieve` from a stub into a working advisory retrieval tool, end-to-end:

```
image ──► frozen CXR encoder ──► query_emb (512-d, normalised)
                                      │
query_emb, k ──► FAISS IndexFlatIP ──► top-k similar VinDr cases
                                      │
                          Observation(result=[case, …], ok=True)
                                      │
                  agent reads cases → picks next SYMBOLIC op
                  (retrieve itself is NOT load-bearing in any Tier-A trace)
```

**Success looks like:** the agent can call `retrieve`, get real similar VinDr cases, and
the deletion test proves those cases are *not* part of any Tier-A faithful trace — yet they
can lift Tier-B answers and guide which graph-op to try next.

---

## 2. Tech stack

- Python (existing repo), `pydantic` frozen models, plain functions.
- `faiss-cpu` — vector index (new dependency, **ask-first** already implied by this spec).
- `open_clip_torch` + `torch` — BiomedCLIP encoder (neural, GPU/optional, integration-only).
- `numpy` — embeddings + the test-time brute-force index.
- Reuse existing: `contracts.py`, `tools/dispatch.py`, `search/tree_search.py`, `pipeline.py`.

---

## 3. Project structure (new files)

```
src/retrieval/
  __init__.py
  encoder.py        # frozen BiomedCLIP image encoder (GPU/optional import)
  index.py          # RagIndex (FAISS) + BruteForceIndex (numpy, test-time) — same .search() shape
  retriever.py      # Retriever: holds index + cases + cached query_emb; set_image / retrieve
  tool.py           # run_retrieve(action, retriever) -> Observation   (the tool wrapper)

data/rag/
  vindr_index.faiss # built by the script (gitignored — large)
  vindr_cases.jsonl # one case per line: {case_id, labels, report?}

scripts/
  build_rag_index.py# embed VinDr-CXR train images -> faiss index + cases.jsonl

tests/
  fixtures/rag_fixture.npz     # ~12 tiny precomputed embeddings + cases (CPU, no GPU)
  test_retrieval_index.py      # BruteForceIndex / RagIndex search correctness + determinism
  test_retrieve_tool.py        # run_retrieve: ok/ok=False, k, graceful degradation
  test_retrieve_faithfulness.py# THE guard tests: retrieve is inert to Tier-A (see §7)
```

Edited (surgical) files: `src/tools/dispatch.py`, `src/search/tree_search.py`,
`src/pipeline.py`, `src/contracts.py` (only if a `RetrievedCase` type is added).

---

## 4. Components

### 4.1 `RetrievedCase` (a plain dict, or a frozen model in `contracts.py`)

```python
class RetrievedCase(BaseModel):
    case_id: str               # = image_id
    labels: List[str]          # distinct gt_finding for the similar case
    report: str = ""           # unique `reason` strings joined — the advisory second opinion
    score: float               # cosine similarity to the query image
    class Config: frozen = True
```
Default to a `dict` if that reads simpler — match whatever the rest of the codebase favours
for tool results (visual tools return plain dicts → I will use a plain dict unless you want
the typed model).

### 4.2 Encoder — `src/retrieval/encoder.py`

```python
def load_encoder(model_name="biomedclip", device="cuda"):
    # import open_clip lazily; return an object with .encode(image) -> np.ndarray (512,)
    # output is L2-normalised. Frozen, eval mode, no grad, deterministic (no sampling).
```
- Neural → **never imported at module top of CPU code**; lazy import inside `load_encoder`.
- Greedy/deterministic: `torch.no_grad()`, `model.eval()`.

### 4.3 Index — `src/retrieval/index.py`

Two implementations, **one duck-typed interface** so tests stay faiss-free and fast:

```python
class RagIndex:          # production
    @classmethod
    def load(cls, faiss_path, cases_path): ...
    def search(self, query_emb, k) -> list[(score, case)]: ...   # FAISS IndexFlatIP

class BruteForceIndex:   # test-time, numpy only, no faiss
    def __init__(self, embeddings, cases): ...
    def search(self, query_emb, k) -> list[(score, case)]: ...   # argsort on cosine
```
Both return the **same** `[(score, case), …]` sorted by descending similarity. Exact search →
deterministic. `BruteForceIndex` must match `RagIndex` on the same vectors (a cross-check test).

### 4.4 Retriever — `src/retrieval/retriever.py`

```python
class Retriever:
    def __init__(self, index, encoder=None):
        self.index = index
        self.encoder = encoder
        self.query_emb = None

    def set_image(self, image):
        # embed the current CXR once; cache it
        self.query_emb = self.encoder.encode(image) if self.encoder else None

    def retrieve(self, k):
        if self.query_emb is None:
            return []
        return [case_with_score(s, c) for s, c in self.index.search(self.query_emb, k)]
```
For tests, `query_emb` can be set directly (skip the encoder), keeping it CPU-only.

### 4.5 Tool wrapper — `src/retrieval/tool.py`

```python
def run_retrieve(action, retriever):
    if retriever is None:
        return Observation(result=None, ok=False)
    k = action.args.get("k", 5)
    cases = retriever.retrieve(k)
    if not cases:
        return Observation(result=None, ok=False)
    return Observation(result=cases, ok=True)
```
Pure given `(action, retriever-state)`. No fact folding — retrieve **adds nothing to
`state_facts`** (it is advisory, not perceptual evidence).

### 4.6 Dispatch wiring — `src/tools/dispatch.py`

`retrieve` is `kind="symbolic"` per the design table but needs the retriever, which the
symbolic path doesn't carry. Route it explicitly and flatly, before the kind branch:

```python
def run_tool(action, facts, dag, img_wh, image=None, detector_fn=None,
             vlm_fn=None, retriever=None):
    if action.tool == "retrieve":
        return run_retrieve(action, retriever)
    if action.kind == "visual":
        return _run_visual(action, image, detector_fn, vlm_fn)
    return run_symbolic(action, facts, dag, img_wh)
```

### 4.7 Search + pipeline wiring

- `tree_search.search(...)` gains `retriever=None`, passed straight into `run_tool`.
  **No fact-folding branch for retrieve** — it only lands in `history`, like every action.
- `pipeline.run(...)` builds the retriever when RAG paths are configured (env
  `RAG_INDEX`, `RAG_CASES`, encoder weights), calls `retriever.set_image(image)`, and
  passes it to `search`. When unset → `retriever=None` → `retrieve` degrades gracefully.

### 4.8 Verifier — **unchanged**

`closure_progress` and `verify` already iterate `history` looking only for the specific
graph-ops per question type (`is_a`, `anatomy_of`, …). A `retrieve` step contributes **no
progress and no penalty** — it is inert by construction. We add **guard tests** (§7) to
lock this property; we do **not** add retrieve-handling code to the verifier.

---

## 5. Corpus build — `scripts/build_rag_index.py`

```
python scripts/build_rag_index.py \
    --vqa data/vindr_cxr_vqa/vqa.json \
    --images data/vindr_cxr_vqa/train \
    --out-index data/rag/vindr_index.faiss --out-cases data/rag/vindr_cases.jsonl
```
Steps: read `vqa.json` → group QA by `image_id` → **keep only image_ids present in the
`train/` split** (skip eval/test images → no leakage) → for each: encode the image, collect
distinct `gt_finding` (labels) and unique `reason` strings (report) → append
`{case_id, labels, report}` to cases.jsonl, embedding to a matrix → build `IndexFlatIP`
on normalised embeddings → save aligned index + cases. GPU. Run once on the server.

---

## 6. Commands

```bash
# CPU unit tests (no GPU, no faiss needed — brute-force index + fixture embeddings)
pytest tests/test_retrieval_index.py tests/test_retrieve_tool.py tests/test_retrieve_faithfulness.py -v

# full suite (must stay green)
pytest tests/

# build the real index (server, GPU, VinDr data)
VINDR_DIR=... python scripts/build_rag_index.py --out-index data/rag/vindr_index.faiss --out-cases data/rag/vindr_cases.jsonl

# GPU integration (encoder + faiss + real index), marked, skipped in CI
pytest tests/test_integration_gpu.py -m gpu -k retrieve -v
```

---

## 7. Testing strategy

Framework: `pytest`, deterministic, CPU-first — same as the rest of the repo. Neural encoder
behind `@pytest.mark.gpu`.

**`test_retrieval_index.py`**
- `BruteForceIndex.search(q, k)` returns exactly `k` cases, descending score.
- Self-query (q = an indexed embedding) → that case ranks #1, score ≈ 1.0.
- Determinism: 100 identical queries → identical ordering.
- `RagIndex` (faiss, if installed) matches `BruteForceIndex` on the same vectors — else skip.

**`test_retrieve_tool.py`**
- `run_retrieve` with a populated retriever → `ok=True`, `len(result) == k`.
- `retriever=None` → `ok=False`. `query_emb=None` → `ok=False`. Empty index → `ok=False`.
- `args={"k": 3}` honoured; default k when omitted.

**`test_retrieve_faithfulness.py` — the load-bearing guarantee**
- **Inert to closure_progress:** a node with a `retrieve` step in history has the *same*
  `closure_progress` as the same node without it.
- **Deletion test (the headline):** take a Tier-A existential trace that *also* contains a
  `retrieve` step; remove the `retrieve` step → answer stays Tier-A "Yes". Remove the
  `is_a` witness → answer flips. Proves retrieve is **not** part of the faithful trace.
- **Never closes Tier-A alone:** a node whose only non-base step is `retrieve` never
  `verify()`s to Tier-A.

**`test_integration_gpu.py` (extend, `-m gpu`)**
- Real BiomedCLIP + real `data/rag` index → `pipeline.run(...)` with a retrieve-using
  MockAgent script returns a `SearchResult`, retrieve observations non-empty.

---

## 8. Code style (one real snippet)

Match `visual.py`: short functions, flat `if`, defensive returns, plain dicts for results.

```python
def run_retrieve(action, retriever):
    if retriever is None:
        return Observation(result=None, ok=False)
    k = action.args.get("k", 5)
    cases = retriever.retrieve(k)
    if not cases:
        return Observation(result=None, ok=False)
    return Observation(result=cases, ok=True)
```
- Lazy-import torch/open_clip/faiss inside loaders only — top of CPU modules stays import-light.
- No randomness, no sampling. Frozen encoder, exact index → deterministic.

---

## 9. Boundaries

**Always**
- Keep retrieve **advisory**: it folds **no** facts into `state_facts`; it never contributes
  to Tier-A closure. The verifier stays untouched and inert to it.
- Degrade gracefully (`ok=False`) when index/encoder/query-embedding is missing.
- Exact, deterministic retrieval (`IndexFlatIP`, frozen encoder, no sampling).
- Run the faithfulness guard tests (§7) before any commit touching retrieve/verifier/search.

**Ask first**
- Adding `faiss-cpu` / `open_clip_torch` to project dependencies.
- Encoder choice if BiomedCLIP is unsuitable (→ MedCLIP) or embedding dim differs.
- Schema change if real paired reports exist (widen `RetrievedCase`).
- Whether retrieve should *ever* gate Tier-B answer selection (today: agent-only signal).

**Never**
- Let a retrieved case become a Tier-A closure step or fold into `state_facts`.
- Use the encoder/retrieval score as the search reward (verifier-as-value stays deterministic).
- Fine-tune the encoder; run it inside `verify()`; introduce sampling/non-determinism.
- Mutate the DAG or base EvaX evidence from retrieval.

---

## 10. Success criteria (testable)

1. `retrieve` returns real top-k VinDr cases for a query image (GPU integration green).
2. `run_retrieve` graceful-degrades to `ok=False` with no retriever/encoder/index (CPU test).
3. **Faithfulness preserved:** deletion of a `retrieve` step never flips a Tier-A answer;
   `closure_progress` is identical with/without retrieve; retrieve alone never reaches Tier-A.
4. Index search is deterministic across 100 runs and `BruteForceIndex` == `RagIndex`.
5. Full existing suite stays green (no regression in core/neural layers).

---

## 11. Open questions (non-blocking — sensible defaults chosen)

1. **Encoder weights**: BiomedCLIP available locally, or let the build script pull from HF on
   the server? (Default: pull from HF on first run.)
2. **k default**: assuming `k=5`, **always return k** (no similarity floor) for now. Say if
   you want a cosine threshold to drop weak matches.
3. ~~Reports vs labels~~ → resolved: use `reason`. ~~Deps~~ → approved. ~~Corpus split~~ →
   resolved: train-only.
```