"""Gold-label validation: run parser on all 17,597 VinDr questions, compare against gold type."""

import json
import pytest
from pathlib import Path
from collections import Counter

from src.question.parser import QuestionParser

VQA_PATH = Path("data/vindr_cxr_vqa/vqa.json")
VINDR_VOCAB = {
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
}

GOLD_TO_QTYPE = {
    "Is_there": "existential",
    "Yes_No": "existential",
    "Where": "relational",
    "Which": "relational",
    "What": "open",
    "How_many": "counting",
}


@pytest.fixture(scope="module")
def vqa_data():
    with open(VQA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def parser():
    return QuestionParser(finding_vocab=VINDR_VOCAB)


@pytest.fixture(scope="module")
def parse_results(vqa_data, parser):
    results = []
    for img in vqa_data:
        for item in img["vqa"]:
            q = parser.parse(item["question"])
            expected_type = GOLD_TO_QTYPE[item["type"]]
            results.append({
                "question": item["question"],
                "gold_type": item["type"],
                "expected_qtype": expected_type,
                "parsed_qtype": q.type,
                "gold_finding": item.get("gt_finding"),
                "parsed_target": q.target,
                "tier": q.parser_tier,
                "confidence": q.parse_confidence,
            })
    return results


def test_type_accuracy_is_100_percent(parse_results):
    wrong = [r for r in parse_results if r["parsed_qtype"] != r["expected_qtype"]]
    total = len(parse_results)
    correct = total - len(wrong)
    accuracy = correct / total * 100

    if wrong:
        sample = wrong[:5]
        details = "\n".join(
            f"  Q: {r['question']} | gold: {r['gold_type']}→{r['expected_qtype']} | got: {r['parsed_qtype']}"
            for r in sample
        )
        pytest.fail(
            f"type_accuracy = {accuracy:.2f}% ({correct}/{total}). "
            f"First mismatches:\n{details}"
        )


def test_target_accuracy(parse_results):
    # Only count types where the finding is named in the question text
    # (Is_there, Yes_No, Where). How_many/What/Which don't mention a finding.
    extractable_types = {"Is_there", "Yes_No", "Where"}
    items = [r for r in parse_results if r["gold_type"] in extractable_types]
    correct = sum(1 for r in items if r["parsed_target"] == r["gold_finding"])
    total = len(items)
    accuracy = correct / total * 100
    print(f"\ntarget_accuracy (extractable types) = {accuracy:.2f}% ({correct}/{total})")
    assert accuracy == 100.0, f"target_accuracy = {accuracy:.2f}%"


def test_open_queries_have_no_target(parse_results):
    open_types = {"How_many", "What", "Which"}
    items = [r for r in parse_results if r["gold_type"] in open_types]
    with_target = [r for r in items if r["parsed_target"] is not None]
    assert len(with_target) == 0, (
        f"{len(with_target)} open queries incorrectly extracted a target"
    )


def test_all_tiers_are_rule(parse_results):
    llm_count = sum(1 for r in parse_results if r["tier"] == "llm")
    assert llm_count == 0, f"{llm_count} questions fell through to LLM/default"


def test_type_accuracy_per_gold_type(parse_results):
    by_type = {}
    for r in parse_results:
        gold = r["gold_type"]
        by_type.setdefault(gold, {"correct": 0, "total": 0})
        by_type[gold]["total"] += 1
        if r["parsed_qtype"] == r["expected_qtype"]:
            by_type[gold]["correct"] += 1

    print("\n--- Type accuracy per gold type ---")
    for t, counts in sorted(by_type.items()):
        acc = counts["correct"] / counts["total"] * 100
        print(f"  {t:12s}: {acc:6.2f}% ({counts['correct']}/{counts['total']})")
        assert acc == 100.0, f"type_accuracy for {t} = {acc:.2f}%"
