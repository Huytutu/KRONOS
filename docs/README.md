# KRONOS — Documentation Index

The project root [README.md](../README.md) is the entry point (overview + how to run).
This folder holds the design and decision records.

## Design

- **[project.tex](project.tex)** — full v4 design document (*verifier-guided multimodal
  tree search*), the authoritative design. Compile with a Vietnamese-capable LaTeX setup
  (VnTeX), or read [project.pdf](project.pdf).

## Specifications (root)

- [v4_core_SPEC.md](../v4_core_SPEC.md) — deterministic core: contracts, symbolic tools,
  verifier, tree search, MockAgent.
- [v4_neural_SPEC.md](../v4_neural_SPEC.md) — neural layer: LLaVA-Med agent, visual tools.
- [ontology_SPEC.md](../ontology_SPEC.md) — the curated ontology DAG (`src/ontology/dag.py`).

## Decision records

- [intent/kronos-v4-multimodal-tree-search.md](intent/kronos-v4-multimodal-tree-search.md)
  — confirmed intent for the current design (what & why).
- [intent/kronos-v3-vlm-agent.md](intent/kronos-v3-vlm-agent.md)
  — prior design (ReAct-on-graph), kept for history.

## Design history

- **v2** — propose-then-verify with a symbolic engine core and three proposer heads.
  Superseded; the v2 design doc and engine spec were removed during the v4 cleanup.
- **v3** — single VLM agent doing ReAct on the graph (linear). Superseded by v4.
- **v4 (current)** — VLM agent searches a *tree* of visual + symbolic actions, guided by a
  deterministic verifier. See [project.tex](project.tex).
