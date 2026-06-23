# How to run the multi-hop eval sweep (GPU)

Everything is built and tested. This is the runbook you execute. Activate the
`medcxr` env first; all commands are from the repo root.

## Prereqs (already verified)
- `data/multihop_qa/qa.jsonl` (300 items) — present.
- MedGemma 4B weights at `weights/medgemma-4b-it`, CUDA, bitsandbytes — present.
- VRAM 5.3 GB free → 4-bit MedGemma fits (the runner uses 4-bit by default).

## Systems
`kronos` (model proposes → KG verifies → trace), `cot`, `zero_shot`,
`react` (same tools, no verifier gate), `single_hop` + `no_reflection` (ablations),
`mock` (KG oracle — sanity only, not a reported result).

## Step 1 — smoke each system on a few items (catch issues fast)
```
python scripts/run_multihop.py --system kronos --limit 5
python scripts/run_multihop.py --system cot    --limit 5
```
Each writes `results/preds_<system>.jsonl`. Eyeball: `kronos` Yes answers should
carry a real `trace`; `cot` traces are `[]`.

## Step 2 — full run, one system at a time
```
for S in kronos cot zero_shot react single_hop no_reflection; do
  python scripts/run_multihop.py --system $S
done
```
~300 items/system; `kronos`/`react` do up to ~3 inferences/item. Expect this to
take a while on the 4050 — run overnight or background each system. (PowerShell:
`foreach ($S in 'kronos','cot','zero_shot','react','single_hop','no_reflection') { python scripts/run_multihop.py --system $S }`)

## Step 3 — grade each system
```
for S in kronos cot zero_shot react single_hop no_reflection; do
  python scripts/eval_multihop.py --qa data/multihop_qa/qa.jsonl \
         --pred results/preds_$S.jsonl --system $S
done
```
Writes `results/multihop_<system>.json` with the five metrics.

## Step 4 — read the table
Compare `binary_accuracy`, `name_accuracy`, `grounding_rate`,
`hallucination_rate`, `load_bearing_rate` across systems. Expected story:
- `kronos`: grounding ≈ 1.0, hallucination ≈ 0 (KG-gated), high accuracy.
- `cot` / `zero_shot`: grounding ≈ 0, higher hallucination (no trace).
- `react`: same tools but ungated → more hallucination than kronos.
- `single_hop` / `no_reflection`: accuracy drops vs full kronos (ablation value).

## Tips
- `--limit N` for quick subset runs; `--out path` to override the predictions path.
- If VRAM is tight, `--no-quantize` is the opposite (more VRAM) — keep the default 4-bit.
- `results/` is gitignored; commit the final `multihop_*.json` table separately if you want it in the paper.
