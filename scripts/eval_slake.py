"""Evaluate KRONOS on SLAKE 1.0 X-Ray VQA subset.

Usage:
  python scripts/eval_slake.py --limit 50
  python scripts/eval_slake.py --limit 0   # full X-Ray subset
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_slake
from src.knowledge.slake_kg import SlakeKG

ONT = ROOT / "data" / "ontology"
DEFAULT_DATA = ROOT / "data" / "Slake1.0" / "test.json"
DEFAULT_IMAGE_DIR = "data/Slake1.0/imgs"
DEFAULT_KG_DIR = ROOT / "data" / "Slake1.0" / "KG"
DEFAULT_WEIGHTS = ROOT / "weights" / "yolov12s_vindr.pt"
DEFAULT_MODEL = ROOT / "weights" / "medgemma-4b-it"


def exact_match(prediction, ground_truth):
    return prediction.strip().lower() == ground_truth.strip().lower()


def _serialize_trace(search_result):
    steps = []
    for action, obs in search_result.path:
        steps.append({
            "tool": action.tool,
            "args": action.args,
            "result": obs.result if isinstance(obs.result, (str, int, float, bool, list, dict, type(None))) else str(obs.result),
            "ok": obs.ok,
        })
    return steps


def init_pipeline(weights, model_path, quantize, use_oracle=False, image_dir=None):
    from src.ontology.dag import OntologyDAG

    dag = OntologyDAG(
        str(ONT / "dag.yaml"),
        str(ONT / "exclusion_lists.yaml"),
        str(ONT / "anatomy_zones.yaml"),
    )

    if use_oracle:
        from src.perception.oracle import SlakeOracle
        detector = SlakeOracle(image_dir or DEFAULT_IMAGE_DIR)
    else:
        from src.perception.detector import Detector
        detector = Detector(str(weights), dag=dag)

    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=str(model_path), quantize=quantize)

    return dag, detector, agent


def run_predictions(items, dag, detector, agent, slake_kg):
    from src.pipeline import run
    results = []
    for i, item in enumerate(items):
        image_path = ROOT / item.image
        if not image_path.exists():
            print(f"  SKIP {item.id}: image not found ({image_path})")
            results.append(None)
            continue
        try:
            result = run(str(image_path), item.question, dag, detector, agent,
                         slake_kg=slake_kg)
            results.append(result)
        except Exception as e:
            print(f"  ERROR {item.id}: {e}")
            results.append(None)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} done")
    return results


def grade(items, search_results):
    details = []
    by_content_type = {}
    by_answer_type = {}
    total_correct = 0

    for item, sr in zip(items, search_results):
        pred = sr.answer if sr else ""
        score = 1 if exact_match(pred, item.answer) else 0
        total_correct += score

        ct = item.meta.get("content_type", "unknown")
        at = item.meta.get("answer_type", "unknown")
        by_content_type.setdefault(ct, []).append(score)
        by_answer_type.setdefault(at, []).append(score)

        detail = {
            "id": item.id, "question": item.question,
            "prediction": pred, "ground_truth": item.answer, "score": score,
            "content_type": ct, "answer_type": at,
        }
        if sr:
            detail["tier"] = sr.tier
            detail["conf"] = sr.conf
            detail["trace"] = _serialize_trace(sr)
        else:
            detail["tier"] = "ABSTAIN"
            detail["conf"] = 0.0
            detail["trace"] = []
        details.append(detail)

    n = len(items)
    def _agg(group):
        return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in group.items()}

    return {
        "n": n,
        "overall_accuracy": total_correct / n if n else 0.0,
        "by_content_type": _agg(by_content_type),
        "by_answer_type": _agg(by_answer_type),
        "details": details,
    }


def print_report(report):
    n = report["n"]
    print(f"\nSLAKE VQA Evaluation — X-Ray (n={n})")
    print(f"  {'overall_accuracy':20s} {report['overall_accuracy']:.3f}")

    print("\n  By content type:")
    for ct, stats in sorted(report["by_content_type"].items()):
        print(f"    {ct:20s} {stats['accuracy']:.3f}  (n={stats['n']})")

    print("\n  By answer type:")
    for at, stats in sorted(report["by_answer_type"].items()):
        print(f"    {at:20s} {stats['accuracy']:.3f}  (n={stats['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="path to test.json")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="image root")
    ap.add_argument("--kg-dir", default=str(DEFAULT_KG_DIR), help="SLAKE KG directory")
    ap.add_argument("--limit", type=int, default=50, help="max questions (0=all)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="MedGemma path")
    ap.add_argument("--out", default=None, help="output report path")
    ap.add_argument("--quantize", action="store_true", help="4-bit quantization")
    ap.add_argument("--detector", choices=["yolo", "oracle"], default="yolo",
                    help="yolo = YOLO model, oracle = ground-truth detection.json")
    args = ap.parse_args()

    print("Loading SLAKE data (X-Ray, English)...")
    items = load_slake(args.data, image_dir=args.image_dir)
    if args.limit > 0:
        items = items[:args.limit]
    print(f"  {len(items)} questions loaded")

    print("Loading SLAKE KG...")
    slake_kg = SlakeKG(args.kg_dir)

    use_oracle = args.detector == "oracle"
    print(f"Initializing pipeline (detector={args.detector})...")
    dag, detector, agent = init_pipeline(
        args.weights, args.model, args.quantize,
        use_oracle=use_oracle, image_dir=args.image_dir,
    )

    print("Running predictions...")
    search_results = run_predictions(items, dag, detector, agent, slake_kg)

    print("Grading (exact match)...")
    report = grade(items, search_results)

    out = Path(args.out) if args.out else ROOT / "results" / "slake_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
