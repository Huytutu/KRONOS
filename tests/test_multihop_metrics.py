"""Tests for multi-hop QA grading — src/eval/multihop_metrics.py.

Uses the real causal KG (auto-loaded). The chain sarcoidosis -> {pleural
thickening, pneumothorax} is a verified shared-cause pair in causal_kg.yaml.
"""
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


def _yes_item(**over):
    item = {
        "id": "y1",
        "finding_a": "Pleural thickening",
        "finding_b": "Pneumothorax",
        "answer": "Yes",
        "gold_causes": ["sarcoidosis"],
        "support_edges": [["sarcoidosis", "Pleural thickening"],
                          ["sarcoidosis", "Pneumothorax"]],
        "hops": 2,
        "single_cause": True,
    }
    item.update(over)
    return item


def _valid_trace():
    return [["sarcoidosis", "Pleural thickening"], ["sarcoidosis", "Pneumothorax"]]


def _no_item():
    return {
        "id": "n1", "finding_a": "Calcification", "finding_b": "Aortic enlargement",
        "answer": "No", "gold_causes": [], "support_edges": [],
        "hops": 2, "single_cause": False,
    }


# --- causal_edge helper on the DAG ---

def test_causal_edge_true_and_false(dag):
    assert dag.causal_edge("sarcoidosis", "Pleural thickening") is True
    assert dag.causal_edge("Pleural thickening", "sarcoidosis") is False  # wrong direction
    assert dag.causal_edge("nonsense", "Pneumothorax") is False


# --- grade_item ---

def test_correct_and_grounded(dag):
    from src.eval.multihop_metrics import grade_item
    pred = {"id": "y1", "answer": "Yes", "cause": "sarcoidosis", "trace": _valid_trace()}
    f = grade_item(_yes_item(), pred, dag)
    assert f["binary_correct"] and f["name_correct"] and f["grounded"]
    assert not f["hallucinated"]


def test_correct_name_but_empty_trace_is_not_grounded(dag):
    from src.eval.multihop_metrics import grade_item
    pred = {"id": "y1", "answer": "Yes", "cause": "sarcoidosis", "trace": []}
    f = grade_item(_yes_item(), pred, dag)
    assert f["name_correct"] and not f["grounded"] and not f["hallucinated"]


def test_wrong_cause_is_hallucinated(dag):
    from src.eval.multihop_metrics import grade_item
    pred = {"id": "y1", "answer": "Yes", "cause": "leukemia", "trace": []}
    f = grade_item(_yes_item(), pred, dag)
    assert not f["name_correct"] and f["hallucinated"] and not f["grounded"]


def test_no_item_handled(dag):
    from src.eval.multihop_metrics import grade_item
    pred = {"id": "n1", "answer": "No", "cause": None, "trace": []}
    f = grade_item(_no_item(), pred, dag)
    assert f["binary_correct"]
    assert not f["name_correct"] and not f["hallucinated"] and not f["grounded"]


# --- aggregate ---

def test_grade_aggregate(dag):
    from src.eval.multihop_metrics import grade
    items = [_yes_item(), _no_item()]
    preds = [
        {"id": "y1", "answer": "Yes", "cause": "sarcoidosis", "trace": _valid_trace()},
        {"id": "n1", "answer": "No", "cause": None, "trace": []},
    ]
    m = grade(items, preds, dag)
    assert m["binary_accuracy"] == 1.0
    assert m["name_accuracy"] == 1.0
    assert m["grounding_rate"] == 1.0
    assert m["hallucination_rate"] == 0.0
    assert m["n"] == 2


# --- deletion / load-bearing test ---

def test_common_causes_nonempty(dag):
    assert dag.common_causes("Pleural thickening", "Pneumothorax")


def test_deletion_holds_when_all_supports_removed(dag):
    from src.eval.multihop_metrics import deletion_holds
    a, b = "Pleural thickening", "Pneumothorax"
    causes = dag.common_causes(a, b)
    support = [[c, a] for c in causes] + [[c, b] for c in causes]
    item = {"finding_a": a, "finding_b": b, "answer": "Yes",
            "single_cause": len(causes) == 1, "support_edges": support}
    assert deletion_holds(item, dag) is True


def test_partial_deletion_does_not_hold(dag):
    from src.eval.multihop_metrics import deletion_holds
    a, b = "Pleural thickening", "Pneumothorax"
    causes = dag.common_causes(a, b)
    if len(causes) < 2:
        pytest.skip("needs a pair with >= 2 common causes")
    support = [[causes[0], a], [causes[0], b]]  # remove only one of several causes
    item = {"finding_a": a, "finding_b": b, "answer": "Yes",
            "single_cause": False, "support_edges": support}
    assert deletion_holds(item, dag) is False


def test_ablation_does_not_mutate_kg(dag):
    before = dag.causal.number_of_edges()
    dag.common_causes("Pleural thickening", "Pneumothorax",
                      removed=[["sarcoidosis", "Pneumothorax"]])
    assert dag.causal.number_of_edges() == before


def test_grade_reports_load_bearing(dag):
    from src.eval.multihop_metrics import grade
    a, b = "Pleural thickening", "Pneumothorax"
    causes = dag.common_causes(a, b)
    support = [[c, a] for c in causes] + [[c, b] for c in causes]
    item = {"id": "y1", "finding_a": a, "finding_b": b, "answer": "Yes",
            "gold_causes": causes, "support_edges": support,
            "hops": 2, "single_cause": True}
    m = grade([item], [{"id": "y1", "answer": "Yes", "cause": causes[0], "trace": []}], dag)
    assert m["load_bearing_rate"] == 1.0
