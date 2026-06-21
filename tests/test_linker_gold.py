"""Gold validation: run linker on every gt_finding in VinDr — must be 100%."""

import json
import pytest
from pathlib import Path
from collections import Counter

from src.linking.linker import ConceptLinker

VQA_PATH = Path("data/vindr_cxr_vqa/vqa.json")
SYNONYMS_PATH = "data/ontology/synonyms.yaml"


@pytest.fixture(scope="module")
def gold_findings():
    with open(VQA_PATH) as f:
        data = json.load(f)
    findings = []
    for img in data:
        for item in img["vqa"]:
            findings.append(item["gt_finding"])
    return findings


@pytest.fixture(scope="module")
def linker():
    return ConceptLinker(SYNONYMS_PATH)


def test_linking_accuracy_is_100_percent(gold_findings, linker):
    total = len(gold_findings)
    unresolved = [f for f in gold_findings if linker.link(f) is None]
    accuracy = (total - len(unresolved)) / total * 100

    print(f"\nlinking_accuracy = {accuracy:.2f}% ({total - len(unresolved)}/{total})")
    if unresolved:
        sample = Counter(unresolved).most_common(5)
        details = "\n".join(f"  {repr(f)}: {c} times" for f, c in sample)
        pytest.fail(f"linking_accuracy = {accuracy:.2f}%. Unresolved:\n{details}")


def test_each_gold_finding_maps_to_itself(gold_findings, linker):
    # In VinDr, every gt_finding is already a canonical name, so the linker
    # should return it unchanged.
    for finding in set(gold_findings):
        assert linker.link(finding) == finding
