"""Evaluate KRONOS on VinDr-CXR VQA dataset.

Usage:
  python scripts/eval_vindr_vqa.py --limit 50
  python scripts/eval_vindr_vqa.py --limit 0   # full dataset
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_vindr_vqa
from src.eval.vindr_vqa_metrics import grade_batch
from src.llm.gemini_client import complete as gemini_complete

ONT = ROOT / "data" / "ontology"
DEFAULT_VQA = ROOT / "data" / "vindr_cxr_vqa" / "vqa.json"
DEFAULT_IMAGE_DIR = "data/vindr_cxr_vqa/train"
DEFAULT_WEIGHTS = ROOT / "weights" / "yolov12s_vindr.pt"
DEFAULT_MODEL = ROOT / "weights" / "medgemma-4b-it"


def init_pipeline(weights, model_path, quantize):
    from src.ontology.dag import OntologyDAG
    from src.perception.detector import Detector

    dag = OntologyDAG(
        str(ONT / "dag.yaml"),
        str(ONT / "exclusion_lists.yaml"),
        str(ONT / "anatomy_zones.yaml"),
    )
    detector = Detector(str(weights), dag=dag)

    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=str(model_path), quantize=quantize)

    return dag, detector, agent


def run_predictions(items, dag, detector, agent):
    from src.pipeline import run
    predictions = []
    for i, item in enumerate(items):
        image_path = ROOT / item.image
        if not image_path.exists():
            print(f"  SKIP {item.id}: image not found ({image_path})")
            predictions.append("")
            continue
        try:
            result = run(str(image_path), item.question, dag, detector, agent)
            predictions.append(result.answer)
        except Exception as e:
            print(f"  ERROR {item.id}: {e}")
            predictions.append("")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} done")
    return predictions


def print_report(report):
    n = report["n"]
    print(f"\nVinDr-CXR VQA Evaluation (n={n})")
    print(f"  {'overall_accuracy':20s} {report['overall_accuracy']:.3f}")

    print("\n  By type:")
    for qtype, stats in sorted(report["by_type"].items()):
        print(f"    {qtype:20s} {stats['accuracy']:.3f}  (n={stats['n']})")

    print("\n  By difficulty:")
    for diff, stats in sorted(report["by_difficulty"].items()):
        print(f"    {diff:20s} {stats['accuracy']:.3f}  (n={stats['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vqa", default=str(DEFAULT_VQA), help="path to vqa.json")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="image directory")
    ap.add_argument("--limit", type=int, default=50, help="max questions (0=all)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="MedGemma path")
    ap.add_argument("--out", default=None, help="output report path")
    ap.add_argument("--quantize", action="store_true", help="4-bit quantization")
    args = ap.parse_args()

    print("Loading VQA data...")
    items = load_vindr_vqa(args.vqa, image_dir=args.image_dir)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"  {len(items)} questions loaded")

    print("Initializing pipeline...")
    dag, detector, agent = init_pipeline(args.weights, args.model, args.quantize)

    print("Running predictions...")
    predictions = run_predictions(items, dag, detector, agent)

    print("Grading with Gemini judge...")
    report = grade_batch(items, predictions, gemini_complete)

    out = Path(args.out) if args.out else ROOT / "results" / "vindr_vqa_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
