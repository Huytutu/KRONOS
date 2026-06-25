"""Tests for SLAKE VQA evaluation pipeline.

All tests are CPU-only (mocked models, mocked APIs).
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Task 1: SlakeKG tests ──

from src.knowledge.slake_kg import SlakeKG


@pytest.fixture
def mini_kg(tmp_path):
    """Create a minimal SLAKE KG on disk."""
    kg_dir = tmp_path / "KG"
    kg_dir.mkdir()
    (kg_dir / "en_disease.csv").write_text(
        "disease#attribute#value\n"
        "Pneumonia#cause#bacterial infection\n"
        "Pneumonia#symptom#cough, fever\n"
        "Pneumonia#location#Lung\n"
        "Pneumonia#treatment#antibiotics\n",
        encoding="utf-8",
    )
    (kg_dir / "en_organ.csv").write_text(
        "organ#attribute#value\n"
        "Lung#function#Breathe\n"
        "Heart#function#Pump blood\n",
        encoding="utf-8",
    )
    (kg_dir / "en_organ_rel.csv").write_text(
        "organ#attribute#value\n"
        "Heart#belong to#Circulatory System\n"
        "Lung#belong to#Respiratory System\n",
        encoding="utf-8",
    )
    return SlakeKG(str(kg_dir))


def test_slake_kg_lookup(mini_kg):
    assert mini_kg.lookup("Pneumonia", "cause") == "bacterial infection"
    assert mini_kg.lookup("Pneumonia", "symptom") == "cough, fever"
    assert mini_kg.lookup("Lung", "function") == "Breathe"
    assert mini_kg.lookup("Heart", "belong to") == "Circulatory System"


def test_slake_kg_case_insensitive(mini_kg):
    assert mini_kg.lookup("pneumonia", "CAUSE") == "bacterial infection"
    assert mini_kg.lookup("LUNG", "Function") == "Breathe"


def test_slake_kg_missing(mini_kg):
    assert mini_kg.lookup("Unknown", "cause") is None
    assert mini_kg.lookup("Pneumonia", "nonexistent") is None


def test_slake_kg_diseases(mini_kg):
    diseases = mini_kg.diseases()
    assert "pneumonia" in diseases


def test_slake_kg_organs(mini_kg):
    organs = mini_kg.organs()
    assert "lung" in organs
    assert "heart" in organs


# ── SlakeOracle tests ──

from src.perception.oracle import SlakeOracle


def test_slake_oracle_detect(tmp_path):
    img_dir = tmp_path / "xmlab1"
    img_dir.mkdir()
    det = [{"Cardiomegaly": [100.0, 200.0, 300.0, 250.0]}]
    (img_dir / "detection.json").write_text(json.dumps(det), encoding="utf-8")

    oracle = SlakeOracle(str(tmp_path))
    facts = oracle.detect(str(img_dir / "source.jpg"))
    assert len(facts) == 1
    assert facts[0].concept == "Cardiomegaly"
    assert facts[0].bbox == (100.0, 200.0, 400.0, 450.0)  # x, y, x+w, y+h
    assert facts[0].conf == 1.0


def test_slake_oracle_no_detection(tmp_path):
    img_dir = tmp_path / "xmlab2"
    img_dir.mkdir()
    oracle = SlakeOracle(str(tmp_path))
    facts = oracle.detect(str(img_dir / "source.jpg"))
    assert facts == []


# ── Task 2: load_slake tests ──

from src.data.loaders import load_slake, QAItem


# ── Task 3: dispatch slake_kg tool test ──

from src.contracts import Action, Observation
from src.tools.dispatch import run_tool


def test_dispatch_slake_kg(mini_kg):
    action = Action(tool="slake_kg", args={"entity": "Pneumonia", "relation": "cause"})
    obs = run_tool(action, facts=[], dag=None, img_wh=None, slake_kg=mini_kg)
    assert obs.ok is True
    assert obs.result == "bacterial infection"


def test_dispatch_slake_kg_missing(mini_kg):
    action = Action(tool="slake_kg", args={"entity": "Unknown", "relation": "cause"})
    obs = run_tool(action, facts=[], dag=None, img_wh=None, slake_kg=mini_kg)
    assert obs.ok is False


def test_dispatch_slake_kg_none():
    action = Action(tool="slake_kg", args={"entity": "Pneumonia", "relation": "cause"})
    obs = run_tool(action, facts=[], dag=None, img_wh=None, slake_kg=None)
    assert obs.ok is False


# ── Task 2: load_slake tests ──

def test_load_slake_filters_xray(tmp_path):
    data = [
        {"img_id": 1, "img_name": "xmlab1/source.jpg", "question": "Q1",
         "answer": "Heart", "q_lang": "en", "location": "Chest",
         "modality": "X-Ray", "answer_type": "OPEN", "base_type": "vqa",
         "content_type": "Organ", "triple": ["vhead", "_", "_"], "qid": 0},
        {"img_id": 2, "img_name": "xmlab2/source.jpg", "question": "Q2",
         "answer": "Liver", "q_lang": "en", "location": "Abdomen",
         "modality": "CT", "answer_type": "OPEN", "base_type": "vqa",
         "content_type": "Organ", "triple": ["vhead", "_", "_"], "qid": 1},
        {"img_id": 3, "img_name": "xmlab3/source.jpg", "question": "Q3",
         "answer": "Yes", "q_lang": "zh", "location": "Chest",
         "modality": "X-Ray", "answer_type": "CLOSED", "base_type": "vqa",
         "content_type": "KG", "triple": ["vhead", "cause", "ktail"], "qid": 2},
    ]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    items = load_slake(str(path), image_dir="data/Slake1.0/imgs")
    assert len(items) == 1
    assert items[0].question == "Q1"
    assert items[0].dataset == "slake"
    assert items[0].meta["content_type"] == "Organ"
    assert items[0].meta["modality"] == "X-Ray"


# ── Task 4: grading + CLI tests ──

from scripts.eval_slake import exact_match, grade


def test_exact_match():
    assert exact_match("Heart", "heart") is True
    assert exact_match("Yes", "No") is False
    assert exact_match(" Lung ", "lung") is True
    assert exact_match("MRI", "mri") is True


def test_grade_aggregation():
    items = [
        QAItem(id="a", dataset="slake", image="x.jpg", question="Q1", answer="Heart",
               meta={"content_type": "Organ", "answer_type": "OPEN"}),
        QAItem(id="b", dataset="slake", image="x.jpg", question="Q2", answer="Yes",
               meta={"content_type": "KG", "answer_type": "CLOSED"}),
        QAItem(id="c", dataset="slake", image="x.jpg", question="Q3", answer="Lung",
               meta={"content_type": "Organ", "answer_type": "OPEN"}),
        QAItem(id="d", dataset="slake", image="x.jpg", question="Q4", answer="No",
               meta={"content_type": "Abnormality", "answer_type": "CLOSED"}),
    ]
    # Mock SearchResults: items 0,1 correct, items 2,3 wrong
    def _sr(answer):
        sr = MagicMock()
        sr.answer = answer
        sr.tier = "A"
        sr.conf = 0.9
        sr.path = []
        return sr

    results = [_sr("Heart"), _sr("Yes"), _sr("Brain"), _sr("Yes")]
    report = grade(items, results)

    assert report["n"] == 4
    assert report["overall_accuracy"] == 0.5
    assert report["by_content_type"]["Organ"]["accuracy"] == 0.5
    assert report["by_content_type"]["KG"]["accuracy"] == 1.0
    assert report["by_answer_type"]["OPEN"]["n"] == 2
    assert report["by_answer_type"]["CLOSED"]["n"] == 2


def test_cli_smoke(tmp_path):
    data = [
        {"img_id": 1, "img_name": "xmlab1/source.jpg", "question": "What organ?",
         "answer": "Heart", "q_lang": "en", "location": "Chest",
         "modality": "X-Ray", "answer_type": "OPEN", "base_type": "vqa",
         "content_type": "Organ", "triple": ["vhead", "_", "_"], "qid": 0},
        {"img_id": 2, "img_name": "xmlab2/source.jpg", "question": "Is there disease?",
         "answer": "Yes", "q_lang": "en", "location": "Chest",
         "modality": "X-Ray", "answer_type": "CLOSED", "base_type": "vqa",
         "content_type": "Abnormality", "triple": ["vhead", "_", "_"], "qid": 1},
    ]
    data_path = tmp_path / "test.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    out_path = tmp_path / "report.json"

    def _sr(answer):
        sr = MagicMock()
        sr.answer = answer
        sr.tier = "A"
        sr.conf = 0.9
        sr.path = []
        return sr

    with patch("scripts.eval_slake.init_pipeline") as mock_init, \
         patch("scripts.eval_slake.run_predictions", return_value=[_sr("Heart"), _sr("Yes")]):
        mock_init.return_value = (MagicMock(), MagicMock(), MagicMock())

        from scripts.eval_slake import main
        import sys
        orig_argv = sys.argv
        sys.argv = ["eval_slake.py",
                     "--data", str(data_path),
                     "--image-dir", str(tmp_path),
                     "--kg-dir", str(tmp_path),
                     "--limit", "2",
                     "--out", str(out_path)]
        try:
            main()
        finally:
            sys.argv = orig_argv

    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["n"] == 2
    assert report["overall_accuracy"] == 1.0
    assert "trace" in report["details"][0]
