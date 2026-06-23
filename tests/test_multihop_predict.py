"""Tests for multi-hop predictors (Phase A — no model).

predict_mock is a deterministic KG oracle used to validate the
predict -> write -> grade pipeline end-to-end without MedGemma.
"""
import json
import pytest
from pathlib import Path
from src.ontology.dag import OntologyDAG

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    return OntologyDAG(
        str(DATA / "dag.yaml"),
        str(DATA / "exclusion_lists.yaml"),
        str(DATA / "anatomy_zones.yaml"),
    )


def _yes_item():
    return {"id": "y1", "finding_a": "Pleural thickening", "finding_b": "Pneumothorax",
            "question": "The chest X-ray shows Pleural thickening and Pneumothorax. "
                        "Could a single condition account for both? If so, name one.",
            "answer": "Yes", "gold_causes": [], "support_edges": [],
            "hops": 2, "single_cause": False}


def _no_item():
    return {"id": "n1", "finding_a": "Calcification", "finding_b": "Aortic enlargement",
            "question": "The chest X-ray shows Calcification and Aortic enlargement. "
                        "Could a single condition account for both? If so, name one.",
            "answer": "No", "gold_causes": [], "support_edges": [],
            "hops": 2, "single_cause": False}


def test_predict_mock_yes(dag):
    from src.eval.predictors import predict_mock
    p = predict_mock(_yes_item(), dag)
    assert p["answer"] == "Yes"
    assert p["cause"] in dag.common_causes("Pleural thickening", "Pneumothorax")
    assert p["trace"] and all(dag.causal_edge(s, t) for s, t in p["trace"])


def test_predict_mock_no(dag):
    from src.eval.predictors import predict_mock
    p = predict_mock(_no_item(), dag)
    assert p["answer"] == "No" and p["cause"] is None and p["trace"] == []


def test_write_predictions_schema_and_grade(dag, tmp_path):
    from src.eval.predictors import predict_mock, write_predictions
    from src.eval.multihop_metrics import grade

    items = [_yes_item(), _no_item()]
    # gold_causes for grading the mock's name correctness
    for it in items:
        it["gold_causes"] = dag.common_causes(it["finding_a"], it["finding_b"])

    out = tmp_path / "preds.jsonl"
    write_predictions(items, lambda it: predict_mock(it, dag), out)

    preds = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert {p["id"] for p in preds} == {"y1", "n1"}
    assert all({"id", "answer", "cause", "trace"} <= set(p) for p in preds)

    m = grade(items, preds, dag)
    assert m["grounding_rate"] == 1.0          # oracle trace is always a real chain
    assert m["hallucination_rate"] == 0.0


# --- prompt + parsing (pure, no model) ---

def test_build_sc_prompt_modes():
    from src.eval.predictors import build_sc_prompt
    item = _yes_item()
    for mode in ("zero_shot", "cot", "react"):
        p = build_sc_prompt(item, mode)
        assert item["question"] in p
    assert "step by step" in build_sc_prompt(item, "cot").lower()


@pytest.mark.parametrize("text,expected", [
    ("Yes, sarcoidosis", ("Yes", "sarcoidosis")),
    ("No", ("No", None)),
    ("Answer: Yes, pulmonary tuberculosis", ("Yes", "pulmonary tuberculosis")),
    ("...long reasoning...\nAnswer: No", ("No", None)),
    ("", ("No", None)),
    ("gibberish with no decision", ("No", None)),
])
def test_parse_yes_no_cause(text, expected):
    from src.eval.predictors import parse_yes_no_cause
    assert parse_yes_no_cause(text) == expected
