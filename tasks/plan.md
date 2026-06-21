# Plan — v4 Neural Layer

> **Status: ✅ COMPLETE.** Both the deterministic core (v4_core_SPEC) and this neural
> layer are implemented and tested (261 CPU tests pass, GPU tests skipped). See
> [README.md](../README.md) for current status and next steps.

Source: [v4_neural_SPEC.md](../v4_neural_SPEC.md)

All unit tests mock VLM/detector (no GPU needed to pass). Integration tests
marked `@pytest.mark.gpu` are written but skipped without GPU.

## Task 1: Visual tools — `src/tools/visual.py`

Implement `inspect`, `re_detect`, `compare` as functions taking Action + image +
detector/vlm → Observation. All VLM/detector calls go through injectable
callables (for mocking). Include fact-folding helper (new facts → merge into
state_facts, dedup by IoU > 0.5).

**Depends on:** v4 core (contracts, symbolic tools)
**Accept:** With mocked VLM/detector responses:
- `inspect` → parses to PerceptualFact.
- `re_detect` → returns facts with bbox mapped to original coords.
- `compare` → parses comparison result.
- Malformed VLM output → `ok=False`.
- Fact-fold deduplicates by IoU.
- Tests in `tests/test_visual_tools.py`.

---

## Task 2: Unified tool dispatch — refactor `src/tools/`

Rename `symbolic.py` logic, create `src/tools/dispatch.py` that routes:
- `action.kind == "symbolic"` → existing symbolic dispatch
- `action.kind == "visual"` → new visual dispatch
Update `tree_search.py` to call `dispatch.run_tool(...)` with optional
`image`/`detector`/`vlm` params. Visual actions gracefully return `ok=False`
when image is None.

**Depends on:** Task 1
**Accept:** Tree search works with both MockAgent (symbolic only, no image) and
visual actions (with mocked image+vlm). All existing 245 tests still pass.
Tests in `tests/test_dispatch.py`.

---

## Task 3: LLaVA-Med Agent — `src/agent/llavamed.py`

Implement `LLaVAMedAgent` satisfying `Agent` Protocol. Model loading via factory
`load_llavamed(path, quantize)`. Prompt construction from
(query, facts, history, tools, reflection). Output parsing: JSON action list or
`Answer[...]`. Malformed → empty list.

**Depends on:** Task 2 (unified dispatch), v4 core (Agent Protocol)
**Accept:** With mocked VLM inference:
- Prompt contains question + facts + tool list + history.
- JSON output `[{"tool":"is_a",...}]` → list of Action.
- `Answer[Yes]` → list with string.
- Malformed → empty list.
- Tests in `tests/test_llavamed_agent.py`.

---

## Task 4: End-to-end pipeline — `src/pipeline.py`

Tie everything: `run(image_path, question, dag, detector, agent)` → SearchResult.
Loads image, runs detector, parses question, sets image on agent, calls search.

**Depends on:** Task 2, Task 3
**Accept:** With MockAgent + oracle detector (existing `src/perception/oracle.py`) →
end-to-end returns SearchResult with tier + path. Test in `tests/test_pipeline.py`.

---

## Task 5: GPU integration tests (write but skip)

Write `tests/test_integration_gpu.py` with `@pytest.mark.gpu`. Tests load real
LLaVA-Med + real YOLO, run on a sample image, verify SearchResult. These tests
are **skipped** without GPU (the user runs them on their server).

**Depends on:** Task 3, Task 4
**Accept:** Tests exist, are properly marked, and are skipped in normal pytest run.
