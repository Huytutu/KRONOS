"""
Reasoning-need diagnostic for VinDr-CXR-VQA.

Classifies each question by the reasoning level KRONOS needs:
  - PERCEPTION_ONLY: answer = direct lookup in detector output (no graph needed)
  - SUBSUMPTION:     answer requires is-a traversal (e.g. "cardiac abnormality?" when fact is "cardiomegaly")
  - RELATIONAL:      answer requires anatomy/laterality mapping (Where/Which)
  - COUNTING:        answer requires aggregating multiple facts
  - OPEN:            free-form listing (What abnormality)
  - NEGATION:        answer requires closed-world check (not present in current data but checked)

The key number: % of questions where reasoning (subsumption/relational/multi-hop)
actually matters vs pure perception lookup. If low (<15-20%), the premise
"multimodal reasoning method" is weak for this dataset.
"""

import json
import re
from collections import Counter
from pathlib import Path

VQA_PATH = Path(__file__).resolve().parent.parent / "data" / "vindr_cxr_vqa" / "vqa.json"

VINDR_FINDINGS = {
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
    "No finding",
}

ABSTRACT_CONCEPTS = {
    "cardiac abnormality", "pulmonary abnormality", "pleural abnormality",
    "vascular abnormality", "abnormality", "finding",
}


def classify_reasoning_need(question, q_type, gt_finding):
    q_lower = question.lower().strip()
    gt_lower = gt_finding.lower().strip() if gt_finding else ""

    if q_type == "How_many":
        return "COUNTING"

    if q_type == "What":
        return "OPEN"

    if q_type in ("Where", "Which"):
        return "RELATIONAL"

    if q_type in ("Yes_No", "Is_there"):
        target = extract_target(q_lower)
        if not target:
            return "PERCEPTION_ONLY"

        target_norm = target.lower()

        is_exact_finding = any(f.lower() == target_norm for f in VINDR_FINDINGS)
        is_abstract = any(a in target_norm for a in ABSTRACT_CONCEPTS)

        has_negation = any(w in q_lower for w in [
            "no finding", "clear", "no abnormality", "rule out",
            "absence", "without", "free of",
        ])

        if has_negation:
            return "NEGATION"
        if is_abstract:
            return "SUBSUMPTION"
        if is_exact_finding:
            return "PERCEPTION_ONLY"

        return "PERCEPTION_ONLY"

    return "PERCEPTION_ONLY"


def extract_target(q_lower):
    patterns = [
        r"does this x-ray show (.+?)[\?\.]",
        r"is there (.+?)[\?\.]",
        r"show (.+?)[\?\.]",
        r"presence of (.+?)[\?\.]",
    ]
    for pat in patterns:
        m = re.search(pat, q_lower)
        if m:
            return m.group(1).strip()
    return None


def main():
    with open(VQA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for img in data:
        for q in img["vqa"]:
            level = classify_reasoning_need(
                q["question"], q["type"], q.get("gt_finding", "")
            )
            results.append({
                "type": q["type"],
                "reasoning": level,
                "question": q["question"],
                "difficulty": q.get("difficulty", ""),
                "gt_finding": q.get("gt_finding", ""),
            })

    total = len(results)
    by_reasoning = Counter(r["reasoning"] for r in results)
    by_type = Counter(r["type"] for r in results)
    cross = Counter((r["type"], r["reasoning"]) for r in results)

    print(f"Total questions: {total}")
    print(f"\n{'='*60}")
    print("REASONING-NEED DISTRIBUTION")
    print(f"{'='*60}")
    for level in ["PERCEPTION_ONLY", "SUBSUMPTION", "NEGATION", "RELATIONAL", "COUNTING", "OPEN"]:
        n = by_reasoning.get(level, 0)
        pct = 100 * n / total
        bar = "#" * int(pct / 2)
        print(f"  {level:<20s} {n:>5d}  ({pct:5.1f}%)  {bar}")

    needs_reasoning = total - by_reasoning.get("PERCEPTION_ONLY", 0)
    pct_reasoning = 100 * needs_reasoning / total
    print(f"\n  NEEDS REASONING:     {needs_reasoning:>5d}  ({pct_reasoning:5.1f}%)")
    print(f"  PERCEPTION ONLY:     {by_reasoning.get('PERCEPTION_ONLY', 0):>5d}  ({100-pct_reasoning:5.1f}%)")

    print(f"\n{'='*60}")
    print("CROSS-TAB: question_type x reasoning_need")
    print(f"{'='*60}")
    print(f"  {'type':<12s}", end="")
    levels = ["PERCEPTION_ONLY", "SUBSUMPTION", "NEGATION", "RELATIONAL", "COUNTING", "OPEN"]
    for lv in levels:
        print(f" {lv[:10]:>10s}", end="")
    print()
    for t in sorted(by_type):
        print(f"  {t:<12s}", end="")
        for lv in levels:
            n = cross.get((t, lv), 0)
            print(f" {n:>10d}", end="")
        print()

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if pct_reasoning >= 50:
        print(f"  {pct_reasoning:.0f}% need reasoning — premise STRONG.")
        print("  Multimodal reasoning method is well-justified on this dataset.")
    elif pct_reasoning >= 20:
        print(f"  {pct_reasoning:.0f}% need reasoning — premise MODERATE.")
        print("  Claim should focus on the reasoning subset; report both.")
    else:
        print(f"  {pct_reasoning:.0f}% need reasoning — premise WEAK.")
        print("  Dataset is perception-dominated; narrow the claim or pick a harder dataset.")


if __name__ == "__main__":
    main()
