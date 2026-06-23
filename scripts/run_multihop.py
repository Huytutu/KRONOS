"""Run a system's predictor over the multi-hop QA -> predictions JSONL.

GPU step (except --system mock): loads frozen MedGemma 4B (4-bit). Run one
system at a time, then grade with scripts/eval_multihop.py.

Usage:
  python scripts/run_multihop.py --system kronos                 # full qa.jsonl
  python scripts/run_multihop.py --system cot --limit 50         # quick subset
Systems: kronos | zero_shot | cot | react | single_hop | no_reflection | mock
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ontology.dag import OntologyDAG
from src.eval import predictors as P

ONT = ROOT / "data" / "ontology"
QA_DEFAULT = ROOT / "data" / "multihop_qa" / "qa.jsonl"


def load_image(item):
    return Image.open(ROOT / item["image"]).convert("RGB")


# system -> predict(item, dag, gen) closure
SYSTEMS = {
    "mock":          lambda it, dag, gen: P.predict_mock(it, dag),
    "zero_shot":     lambda it, dag, gen: P.predict_zero_shot(it, gen, load_image(it)),
    "cot":           lambda it, dag, gen: P.predict_cot(it, gen, load_image(it)),
    "react":         lambda it, dag, gen: P.predict_react(it, dag, gen, load_image(it)),
    "kronos":        lambda it, dag, gen: P.predict_kronos(it, dag, gen, load_image(it)),
    "single_hop":    lambda it, dag, gen: P.predict_kronos(it, dag, gen, load_image(it), multi_hop=False),
    "no_reflection": lambda it, dag, gen: P.predict_kronos(it, dag, gen, load_image(it), reflection=False),
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--qa", default=str(QA_DEFAULT))
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke runs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-quantize", action="store_true", help="load MedGemma in bf16 (needs more VRAM)")
    args = ap.parse_args()

    dag = OntologyDAG(str(ONT / "dag.yaml"), str(ONT / "exclusion_lists.yaml"),
                      str(ONT / "anatomy_zones.yaml"))
    items = load_jsonl(args.qa)
    if args.limit:
        items = items[:args.limit]

    gen = None
    if args.system != "mock":
        from src.agent.medgemma import MedGemmaAgent
        agent = MedGemmaAgent(quantize=not args.no_quantize)
        gen = agent.generate

    predict = SYSTEMS[args.system]
    out = Path(args.out) if args.out else ROOT / "results" / f"preds_{args.system}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            pred = predict(item, dag, gen)
            f.write(json.dumps({"id": item["id"], **pred}) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(items)}")

    print(f"{args.system}: {len(items)} predictions -> {out}")


if __name__ == "__main__":
    main()
