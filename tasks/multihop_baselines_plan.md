# Multi-Hop Baselines + KRONOS Run (T5 / §3.5) — Plan

**Feeds:** the built grading harness (`src/eval/multihop_metrics.py`, `scripts/eval_multihop.py`)
over `data/multihop_qa/qa.jsonl`. **Decision (made):** KRONOS answers shared-cause questions
**model-in-loop** — frozen MedGemma proposes a candidate cause / graph ops; the KG **verifies**
(`causal_edge(D,A)` and `causal_edge(D,B)`) and emits `answer + cause + trace [[D,A],[D,B]]`;
"No" when nothing verifies. The verifier gate ⇒ KRONOS never returns an unsupported cause
(grounding=1.0, hallucination=0 by construction) — that is the contribution vs CoT.

Every system emits a predictions JSONL in the grading schema `{id, answer, cause, trace}`.

---

## Phases (split by GPU dependency)

**Phase A — scaffolding (no GPU, testable now).**
**Phase B — model predictors + sweep (needs medcxr env + MedGemma 4B on RTX 4050; heavy/long).**

## Dependency graph
```
A1 predictors module + MockPredictor ─► A2 prompt+parse (pure)
        │                                     │
        └──────────────►  B1 generate() + zero_shot/cot ─► B2 kronos(model-in-loop)+react
                                                                     │
                                                              B3 ablations ─► B4 full sweep+grade
```

---

## Phase A — no GPU

### A1 — predictors module + MockPredictor
`src/eval/predictors.py`: a thin contract `predict_<system>(item, dag, gen=None) -> {answer, cause,
trace}` and `predict_mock(item, dag)` — a deterministic KG-oracle predictor (uses
`dag.common_causes`) that fills a verified `trace`. Purpose: validate the predict→write→grade loop
end-to-end with **no model**.
- **Acceptance:** `predict_mock` on a Yes item returns answer="Yes", a `cause ∈ gold`, valid
  `trace`; on a No item returns "No". A writer emits schema-valid predictions JSONL.
- **Verify:** `tests/test_multihop_predict.py` green; pipe mock preds through `eval_multihop.py`
  (smoke) → grounding_rate=1.0; full suite green.

### A2 — shared-cause prompt + output parsing (pure)
`build_sc_prompt(item, mode)` for modes zero_shot/cot/react; `parse_yes_no_cause(text) ->
(answer, cause)`. No model.
- **Acceptance:** parser handles "Yes, sarcoidosis", "No", "Answer: Yes — cause: X", junk → ("No"/"" safe default).
- **Verify:** parser unit tests green; full suite green. **Checkpoint → commit Phase A.**

---

## Phase B — GPU (medcxr + MedGemma)

### B1 — `generate()` + zero_shot & cot predictors
Add a small `MedGemmaAgent.generate(prompt)` (wraps `_run_model`, greedy/deterministic). Implement
`predict_zero_shot`, `predict_cot` (trace empty → ungrounded by design).
- **Acceptance:** on 2–3 real items, each returns a parsed {answer, cause}; runs on GPU (quantize
  if VRAM-tight).
- **Verify:** tiny smoke run (e.g. `--limit 3`) writes valid predictions; **manual** (GPU not in CI).

### B2 — kronos (model-in-loop) + react_same_tools
`predict_kronos`: model proposes candidate cause(s); KG **verifies** via `causal_edge` to both
findings; emit verified `trace`, else try `neighbors(A,"caused_by")` and check B; "No" if none
verify. `predict_react`: same proposing over `neighbors`/`find_path` but **no verifier gate**
(accepts the model's named cause as-is → can hallucinate).
- **Acceptance:** on items with a known shared cause, kronos returns Yes + verified trace; on a
  no-common-cause item returns No; react can return an unverified cause.
- **Verify:** smoke on ~5 items each; eyeball traces are real KG edges for kronos.

### B3 — ablations
`single_hop_only` (kronos without multi-hop find_path) and `no_reflection` (without §3.3 loop),
via flags on the kronos predictor.
- **Verify:** smoke runs differ from full kronos on at least some items.

### B4 — full sweep + grade  **(HEAVY / LONG)**
`scripts/run_multihop.py --system <s> [--limit N]` loads MedGemma once, iterates `qa.jsonl`,
writes `results/preds_<s>.jsonl`; then `eval_multihop.py` grades each into
`results/multihop_<s>.json`. Run all systems; assemble a comparison table.
- **Acceptance:** every system has a graded report; kronos grounding_rate ≈ 1.0 and
  hallucination ≈ 0; CoT grounding ≈ 0 with higher hallucination (the contribution).
- **Verify:** reports exist; sanity-check the table. **This step is long — run per-system, possibly
  in background; check in after.**

---

## Checkpoints & stops
- Commit after Phase A (no GPU).
- Each Phase-B predictor: smoke on a few items before any full run.
- **Stop before B4 full sweep** to confirm VRAM/runtime are OK (MedGemma 4B on a 6 GB RTX 4050 may
  need 4-bit qu. — `MedGemmaAgent(quantize=True)`); the sweep is the long pole.

## Boundaries / risks
- Frozen model; deterministic decoding (`do_sample=False`); simple flat code; surgical; suite green.
- VRAM: 4B on 4050 → quantize if needed; reduce `max_new_tokens`.
- Runtime: 300 items × ~5 systems = ~1500 inferences — run per-system, allow `--limit` for smokes,
  consider background.
- Keep `kronos` honestly model-in-loop (model proposes, KG verifies) — not the KG oracle
  (`predict_mock` stays a test fixture, never a reported result).
