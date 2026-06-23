# Multi-Hop QA Subset — Design Spec (§3.4)

**Parent:** `graph_reasoning_SPEC.md` §3.4
**Built by:** the user (this doc is the build spec). **Status:** draft for approval.

---

## 1. Objective

A small, **image-grounded, automatically-gradable** QA set whose questions **require a 2-hop
shared-cause chain** over the RGO `may_cause` graph — the data that demonstrates KRONOS's
multi-hop graph reasoning is more faithful than CoT/ReAct. Single-hop CXR-VQA cannot show this.

**Template (one canonical form):** given a VinDr image annotated with findings **A** and **B**,
ask:

> "The chest X-ray shows **{A}** and **{B}**. Could a single condition account for both? If so, name one."

The reasoning required is **A ← D → B**: a disorder *D* that `may_cause` both findings — a 2-hop
chain found via `neighbors(·, "caused_by")` / `find_path`.

## 2. Ground truth (deterministic, from the KG)

For an ordered-independent pair (A, B), both among the **11 mapped findings** (the 3
observation-only findings are never A/B):

```
common_causes(A, B) = predecessors(A) ∩ predecessors(B)   # disorders D with D may_cause A and D may_cause B
answer  = "Yes" if common_causes else "No"
gold    = common_causes           # set of RGO disorder labels ([] when "No")
hops    = 2                        # edges D→A and D→B
support_edges = [[D, A], [D, B] for D in common_causes]
```

**Framing (state explicitly in the paper):** GT is *faithfulness to a reference differential
graph (RGO)*, **not** medical novelty. Baselines are tested on whether they reproduce these
chains from parametric knowledge — that contrast is the contribution. Negatives ("No") mean *no
common cause in RGO*; RGO may omit a real clinical link — acceptable under this framing, but note it.

## 3. Composition (MVP)

- **~300 items**, all **2-hop**, **~50/50 Yes/No**.
- **Yes** = image has both A and B *and* `common_causes` non-empty.
- **No**  = image has both A and B *and* `common_causes` empty (same question form; GT flips
  naturally — no separate distractor needed).
- **Stratified** across finding pairs and images so no single pair dominates; **fixed seed**.
- Prefer Yes-items with a **single common cause** for the load-bearing subset (§5) — record
  `single_cause: true`.

## 4. Item schema (JSONL, one object per line)

```json
{
  "id": "mh_000123",
  "image": "data/vindr_cxr_vqa/test/<image_id>.png",
  "finding_a": "Pleural thickening",
  "finding_b": "Pneumothorax",
  "question": "The chest X-ray shows pleural thickening and pneumothorax. Could a single condition account for both? If so, name one.",
  "answer": "Yes",
  "gold_causes": ["sarcoidosis"],
  "support_edges": [["sarcoidosis", "pleural thickening"], ["sarcoidosis", "pneumothorax"]],
  "hops": 2,
  "single_cause": true
}
```

## 5. Metrics (all automatic, deterministic)

Per the parent spec's faithfulness-first metric set:

1. **Binary accuracy** — predicted Yes/No vs `answer`.
2. **Name accuracy** (Yes items) — predicted disorder ∈ `gold_causes` (case-insensitive; allow the
   disorder's RGO synonyms).
3. **Grounding rate** — fraction of Yes answers backed by an actual KG chain (a `find_path` /
   `neighbors` trace). KRONOS's Tier-A answers carry this by construction; CoT typically cannot.
4. **Hallucination rate** — predicted a cause ∉ `gold_causes` (unsupported by the KG).
5. **Load-bearing (deletion) test** — on `single_cause` Yes items: delete the item's
   `support_edges` from the KG and re-run. The answer **must flip to "No"** (proves it depended on
   the path, not a guess). Report `load_bearing_rate`.

## 6. Build script (user implements) — `scripts/build_multihop_qa.py`

```
python scripts/build_multihop_qa.py --n 300 --seed 0 \
       --out data/multihop_qa/qa.jsonl
```

Grounded on the VinDr **train** split: `test.csv` has no finding labels (hidden), `train.csv`
does. The model is frozen (nothing trained), so train images are leak-free eval data.

Algorithm:
1. Load `train.csv` → per image, the set of present findings ∩ the 11 mapped findings.
2. Load `OntologyDAG` (auto-loads `causal_kg.yaml`); use causal predecessors for `common_causes`.
3. For each image, for each unordered finding pair → build a candidate item (Yes/No + support).
4. Split candidates into Yes / No pools; **stratified-sample** to ~n, ~50/50, fixed seed.
5. Write JSONL + a summary (counts, Yes/No, per-pair distribution, `single_cause` count).

Helper already available: `dag.causal_neighbors(name, "caused_by")` returns a finding's causes;
intersect two findings' cause-sets for `common_causes`.

## 7. Reproducibility & boundaries

**Always**
- GT strictly from the KG; every Yes item records real `support_edges`. No fabricated links.
- Only the 11 mapped findings as A/B; image must actually contain both (grounded).
- Deterministic: fixed seed, checked-in script + output JSONL (small) + summary.

**Never**
- Use observation-only findings (Consolidation/Infiltration/Other lesion) as A/B.
- Hand-author gold answers (keep the metric automatic) for the MVP.
- Let one finding pair or image dominate the sample.

## 8. Open checks before/at build

- **Feasibility count:** ✅ resolved — train split yields 1109 Yes / 16568 No candidates;
  generated 150/150 balanced, 117 single-cause Yes (deletion test holds on all 117).
- **VinDr annotation source:** ✅ resolved — `train.csv` (test labels are hidden).
- **Name grading:** finalize the synonym set per gold disorder (reuse RGO labels; extend if needed).

## 9. Acceptance criteria (definition of done for the data)

- [ ] `data/multihop_qa/test.jsonl` (~300 items, ~50/50), each schema-valid and image-grounded.
- [ ] Every Yes item has ≥1 `support_edges` chain present in `causal_kg.yaml`.
- [ ] Deletion test: removing `support_edges` flips GT to "No" for all `single_cause` items.
- [ ] Summary report committed alongside the data; generation is seed-reproducible.
