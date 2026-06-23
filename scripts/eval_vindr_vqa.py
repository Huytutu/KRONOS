"""Evaluate KRONOS on VinDr-CXR-VQA.

Usage:
    python scripts/eval_vindr_vqa.py --results results.jsonl
    python scripts/eval_vindr_vqa.py --run --weights weights/yolov12s_vindr.pt

results.jsonl format (one line per question):
    {"image_id": "...", "question": "...", "type": "Is_there",
     "gold_answer": "Yes", "pred_answer": "Yes", "tier": "A",
     "pred_conf": 0.95, "gt_finding": "...", "gt_location": "..."}
"""
import json
import re
import argparse
from collections import defaultdict


def extract_yes_no(text):
    """Extract Yes/No from free-text answer."""
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None


def extract_count(text):
    """Extract integer count from free-text answer."""
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    text = text.strip().lower()
    for word, num in word_to_num.items():
        if word in text:
            return num
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[0])
    return None


def extract_laterality(text):
    """Extract laterality from free-text answer."""
    text = text.strip().lower()
    if "bilateral" in text or "both" in text:
        return "bilateral"
    if "right" in text:
        return "right"
    if "left" in text:
        return "left"
    if "central" in text or "midline" in text:
        return "midline"
    return None


BINARY_TYPES = {"Is_there", "Yes_No"}
COUNT_TYPES = {"How_many"}
LATERAL_TYPES = {"Where", "Which"}
OPEN_TYPES = {"What"}


def compare_one(gold_answer, pred_answer, q_type):
    """Compare gold vs predicted answer. Returns True/False/None (None = can't compare)."""
    if q_type in BINARY_TYPES:
        g = extract_yes_no(gold_answer)
        p = extract_yes_no(pred_answer)
        if g is None or p is None:
            return None
        return g == p

    if q_type in COUNT_TYPES:
        g = extract_count(gold_answer)
        p = extract_count(pred_answer)
        if g is None or p is None:
            return None
        return g == p

    if q_type in LATERAL_TYPES:
        g = extract_laterality(gold_answer)
        p = extract_laterality(pred_answer)
        if g is None or p is None:
            return None
        return g == p

    return None


def compute_binary_metrics(results):
    """Precision, Recall, F1 for binary (Yes/No) questions only."""
    tp = fp = fn = tn = 0
    for r in results:
        if r["type"] not in BINARY_TYPES:
            continue
        if r["tier"] == "ABSTAIN":
            continue

        gold = extract_yes_no(r["gold_answer"])
        pred = extract_yes_no(r["pred_answer"])
        if gold is None or pred is None:
            continue

        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "yes" and pred == "no":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_metrics(results):
    """Compute all metrics from a list of result dicts."""
    by_type = defaultdict(list)
    by_tier = defaultdict(list)

    for r in results:
        by_type[r["type"]].append(r)
        by_tier[r["tier"]].append(r)

    # --- Per-type accuracy ---
    type_acc = {}
    for qtype, items in by_type.items():
        answered = [r for r in items if r["tier"] != "ABSTAIN"]
        if not answered:
            type_acc[qtype] = {"accuracy": None, "answered": 0, "total": len(items)}
            continue

        correct = 0
        comparable = 0
        for r in answered:
            result = compare_one(r["gold_answer"], r["pred_answer"], qtype)
            if result is not None:
                comparable += 1
                if result:
                    correct += 1

        type_acc[qtype] = {
            "accuracy": round(correct / comparable, 4) if comparable > 0 else None,
            "correct": correct,
            "comparable": comparable,
            "answered": len(answered),
            "total": len(items),
            "abstained": len(items) - len(answered),
        }

    # --- Overall accuracy (structured types only) ---
    structured = [r for r in results if r["type"] not in OPEN_TYPES and r["tier"] != "ABSTAIN"]
    correct_all = 0
    comparable_all = 0
    for r in structured:
        result = compare_one(r["gold_answer"], r["pred_answer"], r["type"])
        if result is not None:
            comparable_all += 1
            if result:
                correct_all += 1

    overall_acc = round(correct_all / comparable_all, 4) if comparable_all > 0 else None

    # --- Selective accuracy + coverage ---
    total_structured = len([r for r in results if r["type"] not in OPEN_TYPES])
    answered_structured = len(structured)
    coverage = round(answered_structured / total_structured, 4) if total_structured > 0 else 0

    # --- Tier distribution ---
    tier_dist = {t: len(items) for t, items in by_tier.items()}

    # --- Binary P/R/F1 ---
    binary = compute_binary_metrics(results)

    return {
        "overall_accuracy": overall_acc,
        "selective_accuracy": overall_acc,
        "coverage": coverage,
        "per_type": type_acc,
        "binary_prf": binary,
        "tier_distribution": tier_dist,
        "total_questions": len(results),
    }


def print_report(metrics):
    """Print a readable report."""
    print("=" * 60)
    print("KRONOS Evaluation — VinDr-CXR-VQA")
    print("=" * 60)

    print(f"\nTotal questions:      {metrics['total_questions']}")
    print(f"Overall accuracy:     {metrics['overall_accuracy']}")
    print(f"Selective accuracy:   {metrics['selective_accuracy']}")
    print(f"Coverage:             {metrics['coverage']}")

    print(f"\nTier distribution:    {metrics['tier_distribution']}")

    print("\n--- Per-type accuracy ---")
    for qtype, info in sorted(metrics["per_type"].items()):
        acc = info["accuracy"]
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"  {qtype:12s}  acc={acc_str}  "
              f"answered={info['answered']}/{info['total']}  "
              f"abstained={info.get('abstained', 0)}")

    b = metrics["binary_prf"]
    print(f"\n--- Binary (Yes/No) metrics ---")
    print(f"  Precision: {b['precision']:.4f}  Recall: {b['recall']:.4f}  F1: {b['f1']:.4f}")
    print(f"  TP={b['tp']}  FP={b['fp']}  FN={b['fn']}  TN={b['tn']}")


# --- Generate results by running KRONOS pipeline ---

def generate_results(vqa_path, weights_path, dag_dir="data/ontology"):
    """Run KRONOS on all VQA questions and return results list."""
    import sys
    sys.path.insert(0, ".")
    from src.pipeline import run, VINDR_FINDINGS
    from src.perception.detector import Detector
    from src.ontology.dag import OntologyDAG
    from src.agent.mock import MockAgent
    from pathlib import Path

    # All three files are required: dag.yaml alone leaves exclusion lists and
    # anatomy zones unloaded, which makes every negation and "where" question abstain.
    dag = OntologyDAG(
        f"{dag_dir}/dag.yaml",
        f"{dag_dir}/exclusion_lists.yaml",
        f"{dag_dir}/anatomy_zones.yaml",
    )
    detector = Detector(weights_path, dag=dag)
    agent = MockAgent()

    with open(vqa_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for i, item in enumerate(dataset):
        image_id = item["image_id"]
        image_path = Path("data/vindr_cxr_vqa/train") / f"{image_id}.png"
        if not image_path.exists():
            image_path = Path("data/vindr_cxr_vqa/test") / f"{image_id}.png"
        if not image_path.exists():
            continue

        for qa in item["vqa"]:
            try:
                result = run(
                    str(image_path), qa["question"], dag, detector, agent
                )
            except Exception as e:
                result = None

            gold = extract_yes_no(qa["answer"]) or qa["answer"]

            results.append({
                "image_id": image_id,
                "question": qa["question"],
                "type": qa["type"],
                "gold_answer": qa["answer"],
                "pred_answer": result.answer if result else "",
                "tier": result.tier if result else "ABSTAIN",
                "pred_conf": result.conf if result else 0.0,
                "gt_finding": qa["gt_finding"],
                "gt_location": qa["gt_location"],
            })

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(dataset)} images...")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KRONOS on VinDr-CXR-VQA")
    parser.add_argument("--results", type=str, help="Path to results.jsonl (pre-computed)")
    parser.add_argument("--run", action="store_true", help="Run pipeline on dataset")
    parser.add_argument("--weights", type=str, default="weights/yolov12s_vindr.pt")
    parser.add_argument("--vqa", type=str, default="data/vindr_cxr_vqa/vqa.json")
    parser.add_argument("--output", type=str, default="results.jsonl")
    args = parser.parse_args()

    if args.results:
        with open(args.results, "r") as f:
            results = [json.loads(line) for line in f]
    elif args.run:
        results = generate_results(args.vqa, args.weights)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)} results to {args.output}")
    else:
        parser.error("Specify --results or --run")

    metrics = compute_metrics(results)
    print_report(metrics)
