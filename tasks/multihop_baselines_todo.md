# Multi-Hop Baselines + KRONOS Run (T5) — Task List

Source: `tasks/multihop_baselines_plan.md`.

## Phase A — no GPU (build now)
### A1 — predictors module + MockPredictor  [depends: none]
- [ ] `src/eval/predictors.py`: `predict_<system>` contract + `predict_mock(item, dag)` (KG oracle, verified trace).
- [ ] Prediction-writer (JSONL in grading schema).
- [ ] Tests `tests/test_multihop_predict.py`: mock Yes/No correct + schema; mock preds → `eval_multihop` grounding=1.0.
- [ ] **Verify:** new + full suite green. **Checkpoint → commit.**

### A2 — prompt + parsing (pure)  [depends: A1]
- [ ] `build_sc_prompt(item, mode)` (zero_shot/cot/react); `parse_yes_no_cause(text)`.
- [ ] Tests: parser handles Yes+name / No / messy / junk-safe-default.
- [ ] **Verify:** new + full suite green. **Checkpoint → commit Phase A.**

## Phase B — GPU (medcxr + MedGemma; heavy)
### B1 — generate() + zero_shot & cot  [depends: A2]
- [ ] `MedGemmaAgent.generate(prompt)` (greedy). `predict_zero_shot`, `predict_cot` (trace empty).
- [ ] Smoke `--limit 3` writes valid predictions. (manual; GPU)

### B2 — kronos (model-in-loop) + react_same_tools  [depends: B1]
- [ ] `predict_kronos`: model proposes cause → KG verifies via `causal_edge` → trace; fallback `neighbors(A,'caused_by')`∩B; No if none.
- [ ] `predict_react`: same tools, NO verifier gate.
- [ ] Smoke ~5 items each; kronos traces are real KG edges.

### B3 — ablations  [depends: B2]
- [ ] `single_hop_only`, `no_reflection` flags on kronos.
- [ ] Smoke differs from full kronos.

### B4 — full sweep + grade  [depends: B3]  **HEAVY/LONG**
- [ ] `scripts/run_multihop.py --system <s>` → `results/preds_<s>.jsonl`; grade → `results/multihop_<s>.json`.
- [ ] **STOP before full sweep** to confirm VRAM/runtime (quantize if needed); run per-system / background.
- [ ] Assemble comparison table; sanity-check (kronos grounded≈1, CoT grounded≈0).
