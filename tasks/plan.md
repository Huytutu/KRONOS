# Plan — E_rag (the `retrieve` tool)

> Source: [E_rag_SPEC.md](../E_rag_SPEC.md)
>
> Build the `retrieve` tool end-to-end: FAISS index over VinDr-CXR train cases
> (embedded by frozen BiomedCLIP), wired into dispatch/search/pipeline.
> Advisory only — never closes Tier-A; faithfulness guard tests prove it.
>
> CPU tests use a numpy BruteForceIndex + fixture embeddings (no GPU, no faiss).
> Neural encoder + real index behind `@pytest.mark.gpu`.

---

## Task 1: Index layer — `src/retrieval/index.py` + fixture + tests

Create `src/retrieval/` package. Implement `BruteForceIndex` (numpy, CPU) and
`RagIndex` (faiss, optional import). Both share `.search(query_emb, k) →
[(score, case_dict), ...]` sorted descending. Create a tiny fixture
(`tests/fixtures/rag_fixture.npz`) with ~12 synthetic 512-d normalised embeddings
+ matching case dicts.

**Depends on:** nothing
**Files:** `src/retrieval/__init__.py`, `src/retrieval/index.py`,
  `tests/fixtures/rag_fixture.npz`, `tests/test_retrieval_index.py`
**Accept:**
- `BruteForceIndex.search(q, k)` returns exactly k results, descending score.
- Self-query (q = indexed embedding) → that case ranks #1, score ≈ 1.0.
- Determinism: 100 identical queries → identical ordering.
- `RagIndex` matches `BruteForceIndex` on same vectors (skip if faiss not installed).
- All tests pass: `pytest tests/test_retrieval_index.py -v`

---

## Task 2: Retriever + tool wrapper — `src/retrieval/retriever.py`, `src/retrieval/tool.py` + tests

`Retriever` holds an index + cached query_emb. `set_image(emb)` caches the
embedding (encoder not wired yet — raw emb for now). `retrieve(k)` → list of
case dicts with score. `run_retrieve(action, retriever) → Observation`.

**Depends on:** Task 1
**Files:** `src/retrieval/retriever.py`, `src/retrieval/tool.py`,
  `tests/test_retrieve_tool.py`
**Accept:**
- `run_retrieve` with populated retriever → `ok=True`, `len(result) == k`.
- `retriever=None` → `ok=False`.
- `query_emb=None` (not set) → `ok=False`.
- Empty index → `ok=False`.
- `args={"k": 3}` honoured; default k=5 when omitted.
- All tests pass: `pytest tests/test_retrieve_tool.py -v`

---

## Task 3: Wire into dispatch / search / pipeline

Route `retrieve` in `dispatch.run_tool` (flat early-return before kind branch).
Thread `retriever=None` through `search()` and `pipeline.run()`. No fact-folding
for retrieve — result lands in history only. When retriever is None, retrieve
degrades gracefully (`ok=False`).

**Depends on:** Task 2
**Files:** `src/tools/dispatch.py`, `src/tools/symbolic.py`,
  `src/search/tree_search.py`, `src/pipeline.py`
**Accept:**
- `retrieve` action dispatched correctly through search → tool → Observation.
- `retriever=None` → graceful `ok=False` (no crash).
- No `state_facts` changes from retrieve (advisory only).
- **Full existing test suite stays green:** `pytest tests/ -v`

---

## Task 4: Faithfulness guard tests

The load-bearing guarantee: retrieve is inert to Tier-A. Tests use MockAgent +
existing DAG fixtures.

**Depends on:** Task 3
**Files:** `tests/test_retrieve_faithfulness.py`
**Accept:**
- `closure_progress` identical with/without a retrieve step in history.
- Deletion test: remove retrieve from a Tier-A trace → answer unchanged;
  remove the `is_a` witness → answer flips.
- A node whose only action is `retrieve` never verifies to Tier-A.
- All tests pass: `pytest tests/test_retrieve_faithfulness.py -v`

---

## Task 5: Encoder + build script + GPU integration

BiomedCLIP encoder (`src/retrieval/encoder.py`), corpus build script
(`scripts/build_rag_index.py`), and GPU integration test extension.
Lazy imports for torch/open_clip. Build script reads `vqa.json`, filters
to train-split images, encodes, saves FAISS index + cases.jsonl.

**Depends on:** Tasks 1–4
**Files:** `src/retrieval/encoder.py`, `scripts/build_rag_index.py`,
  `tests/test_integration_gpu.py`
**Accept:**
- `load_encoder()` returns object with `.encode(image) → np.ndarray(512,)`.
- Build script produces `vindr_index.faiss` + `vindr_cases.jsonl` from train split.
- GPU integration test: real encoder + real index → pipeline returns SearchResult
  with non-empty retrieve observations.
- Tests marked `@pytest.mark.gpu`, skipped without GPU.
- **Full suite still green:** `pytest tests/ -v`
