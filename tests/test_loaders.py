"""Tests for dataset loaders — every dataset maps to the common QAItem shape."""
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _skip_if_missing(path):
    if not path.exists():
        pytest.skip(f"{path} not present")


# --- common shape ---

def test_qaitem_core_fields():
    from src.data.loaders import QAItem
    it = QAItem(id="x", dataset="multihop", image="a.png",
                question="q?", answer="Yes")
    assert it.id == "x" and it.dataset == "multihop"
    assert it.image == "a.png" and it.question == "q?" and it.answer == "Yes"
    assert it.meta == {}            # meta defaults to empty, never shared between items


def test_registry_lists_all():
    from src.data.loaders import LOADERS
    assert set(LOADERS) == {"multihop", "vindr_vqa", "slake", "chestagentbench"}


# --- multihop ---

def test_load_multihop():
    from src.data.loaders import load_multihop
    path = ROOT / "data" / "multihop_qa" / "qa.jsonl"
    _skip_if_missing(path)
    items = load_multihop(path)
    assert items
    it = items[0]
    assert it.dataset == "multihop"
    assert it.image.endswith(".png")
    assert it.answer in ("Yes", "No")
    assert "finding_a" in it.meta and "finding_b" in it.meta


# --- VinDr-CXR-VQA ---

def test_load_vindr_vqa_flattens_questions():
    from src.data.loaders import load_vindr_vqa
    path = ROOT / "data" / "vindr_cxr_vqa" / "vqa.json"
    _skip_if_missing(path)
    items = load_vindr_vqa(path)
    assert items
    it = items[0]
    assert it.dataset == "vindr_vqa"
    assert "/vindr_cxr_vqa/train/" in it.image and it.image.endswith(".png")
    assert it.question and it.answer
    assert "gt_finding" in it.meta
    # one study contributes several questions → ids are unique per question
    assert len({i.id for i in items}) == len(items)


# --- ChestAgentBench ---

def test_load_chestagentbench():
    from src.data.loaders import load_chestagentbench
    path = ROOT / "data" / "chestagentbench" / "metadata.jsonl"
    _skip_if_missing(path)
    items = load_chestagentbench(path)
    assert items
    it = items[0]
    assert it.dataset == "chestagentbench"
    assert it.image.startswith("data/chestagentbench/figures/")
    assert it.answer in list("ABCDEF")
    assert it.meta["images"]            # full image list kept in meta
