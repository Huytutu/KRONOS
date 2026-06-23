"""Grade a system's predictions on the multi-hop shared-cause QA subset.

Usage:
  python scripts/eval_multihop.py --qa data/multihop_qa/qa.jsonl \
         --pred results/preds_<system>.jsonl --system <system>

Reads QA items + predictions (JSONL), grades against the causal KG, prints the
five metrics, and writes results/multihop_<system>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ontology.dag import OntologyDAG
from src.eval.multihop_metrics import grade

ONT = ROOT / "data" / "ontology"
METRIC_KEYS = ["binary_accuracy", "name_accuracy", "grounding_rate",
               "hallucination_rate", "load_bearing_rate"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", required=True, help="QA items JSONL")
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--system", default="system", help="system name (for the report)")
    ap.add_argument("--out", default=None, help="report path (default results/multihop_<system>.json)")
    args = ap.parse_args()

    dag = OntologyDAG(str(ONT / "dag.yaml"), str(ONT / "exclusion_lists.yaml"),
                      str(ONT / "anatomy_zones.yaml"))
    metrics = grade(load_jsonl(args.qa), load_jsonl(args.pred), dag)

    out = Path(args.out) if args.out else ROOT / "results" / f"multihop_{args.system}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"system": args.system, **metrics}, indent=2), encoding="utf-8")

    print(f"System: {args.system}  (n={metrics['n']})")
    for k in METRIC_KEYS:
        print(f"  {k:20s} {metrics[k]:.3f}")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
