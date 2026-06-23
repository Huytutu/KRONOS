"""Generate the multi-hop shared-cause QA subset (multihop_qa_SPEC.md).

Grounded on VinDr TRAIN images — train.csv has per-image findings; test labels
are hidden. The model is frozen (no training), so train images are leak-free eval.

For each train image with >=2 mapped findings, each finding pair (A, B) becomes:
  "The chest X-ray shows A and B. Could a single condition account for both? ..."
  Yes + the shared cause(s) when RGO has a disorder D with D may_cause A and D may_cause B,
  else No. Then a stratified, ~50/50 sample of size --n is written.

Run: python scripts/build_multihop_qa.py --n 300 --seed 0
"""
import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ontology.dag import OntologyDAG

VQA = ROOT / "data" / "vindr_cxr_vqa"
ONT = ROOT / "data" / "ontology"
TEMPLATE = ("The chest X-ray shows {a} and {b}. Could a single condition account "
            "for both? If so, name one.")


def mapped_findings():
    """The 11 findings present in the causal graph (seeds of causal_kg.yaml)."""
    seeds = yaml.safe_load((ONT / "causal_kg.yaml").read_text(encoding="utf-8"))["seeds"]
    return set(seeds)


def per_image_findings(allowed):
    """image_id -> sorted list of mapped findings present (>=2 only)."""
    df = pd.read_csv(VQA / "train.csv")
    df = df[df["class_name"].isin(allowed)]
    groups = df.groupby("image_id")["class_name"].apply(lambda s: sorted(set(s)))
    return {img: f for img, f in groups.items() if len(f) >= 2}


def build_candidates(image_findings, dag):
    items = []
    for image_id, findings in image_findings.items():
        for a, b in combinations(findings, 2):
            causes = dag.common_causes(a, b)
            items.append({
                "image": f"data/vindr_cxr_vqa/train/{image_id}.png",
                "finding_a": a,
                "finding_b": b,
                "question": TEMPLATE.format(a=a, b=b),
                "answer": "Yes" if causes else "No",
                "gold_causes": causes,
                "support_edges": [[c, a] for c in causes] + [[c, b] for c in causes],
                "hops": 2,
                "single_cause": len(causes) == 1,
            })
    return items


def stratified_sample(items, k, seed):
    """Round-robin across (finding-pair) buckets so no pair dominates the sample."""
    rng = random.Random(seed)
    buckets = {}
    for it in items:
        buckets.setdefault((it["finding_a"], it["finding_b"]), []).append(it)
    for b in buckets.values():
        rng.shuffle(b)

    order = sorted(buckets)
    chosen, i = [], 0
    while len(chosen) < k and any(buckets[p] for p in order):
        p = order[i % len(order)]
        if buckets[p]:
            chosen.append(buckets[p].pop())
        i += 1
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "multihop_qa" / "qa.jsonl"))
    args = ap.parse_args()

    dag = OntologyDAG(str(ONT / "dag.yaml"), str(ONT / "exclusion_lists.yaml"),
                      str(ONT / "anatomy_zones.yaml"))
    candidates = build_candidates(per_image_findings(mapped_findings()), dag)
    yes = [c for c in candidates if c["answer"] == "Yes"]
    no = [c for c in candidates if c["answer"] == "No"]

    half = args.n // 2
    chosen = stratified_sample(yes, half, args.seed) + stratified_sample(no, half, args.seed)
    random.Random(args.seed).shuffle(chosen)
    for i, it in enumerate(chosen):
        it["id"] = f"mh_{i:05d}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in chosen:
            f.write(json.dumps(it) + "\n")

    n_yes = sum(c["answer"] == "Yes" for c in chosen)
    n_single = sum(c["answer"] == "Yes" and c["single_cause"] for c in chosen)
    print(f"candidates: {len(candidates)} (yes={len(yes)}, no={len(no)})")
    print(f"written {len(chosen)} -> {out}")
    print(f"  yes={n_yes}, no={len(chosen) - n_yes}, single_cause-yes={n_single}")


if __name__ == "__main__":
    main()
