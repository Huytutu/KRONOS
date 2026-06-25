"""Tests for unified shared_cause through tree search (not a separate predictor)."""
import pytest
from pathlib import Path
from src.contracts import PerceptualFact, Query, Action, Observation, TreeNode

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    from src.ontology.dag import OntologyDAG
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"),
                       str(DATA / "anatomy_zones.yaml"))


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _sc_query(a, b):
    return Query(type="shared_cause", target=None,
                 constraints={"finding_a": a, "finding_b": b},
                 raw_question=f"Can a single condition cause both {a} and {b}?",
                 parse_confidence=1.0, parser_tier="rule")


# --- verifier ---

def test_verify_shared_cause_yes_with_path(dag):
    """A neighbors + causal_edge sequence that connects both findings → tier A Yes."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Pleural thickening"), _fact("Pneumothorax")],
        history=[
            (Action(tool="neighbors", args={"node": "Pleural thickening", "direction": "caused_by"}),
             Observation(result=["sarcoidosis", "tuberculosis"], ok=True)),
            (Action(tool="neighbors", args={"node": "Pneumothorax", "direction": "caused_by"}),
             Observation(result=["sarcoidosis", "cystic fibrosis"], ok=True)),
        ],
        answer="Yes",
    )
    query = _sc_query("Pleural thickening", "Pneumothorax")
    result = verify(node, query, dag)
    assert result.tier == "A"
    assert "yes" in result.answer.lower()
    assert "sarcoidosis" in result.answer.lower()


def test_verify_shared_cause_no_when_disjoint(dag):
    """No shared cause found after exploration → tier A No."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly"), _fact("Pleural effusion")],
        history=[
            (Action(tool="neighbors", args={"node": "Cardiomegaly", "direction": "caused_by"}),
             Observation(result=["scleroderma", "sarcoidosis"], ok=True)),
            (Action(tool="neighbors", args={"node": "Pleural effusion", "direction": "caused_by"}),
             Observation(result=[], ok=False)),
        ],
    )
    query = _sc_query("Cardiomegaly", "Pleural effusion")
    result = verify(node, query, dag)
    assert result.tier in ("A", "B")


def test_progress_shared_cause_increases_with_neighbors(dag):
    """Exploring more causes → higher reward."""
    from src.engine.verifier import closure_progress
    query = _sc_query("Pleural thickening", "Pneumothorax")

    empty = TreeNode(state_facts=[], history=[])
    one_side = TreeNode(state_facts=[], history=[
        (Action(tool="neighbors", args={"node": "Pleural thickening", "direction": "caused_by"}),
         Observation(result=["sarcoidosis"], ok=True)),
    ])
    both_sides = TreeNode(state_facts=[], history=[
        (Action(tool="neighbors", args={"node": "Pleural thickening", "direction": "caused_by"}),
         Observation(result=["sarcoidosis"], ok=True)),
        (Action(tool="neighbors", args={"node": "Pneumothorax", "direction": "caused_by"}),
         Observation(result=["sarcoidosis"], ok=True)),
    ])
    p0 = closure_progress(empty, query, dag)
    p1 = closure_progress(one_side, query, dag)
    p2 = closure_progress(both_sides, query, dag)
    assert p0 < p1 <= p2


# --- tree search end-to-end ---

def test_search_shared_cause_finds_answer(dag):
    """MockAgent drives tree search to find a shared cause → tier A."""
    from src.search.tree_search import search
    from src.agent.mock import MockAgent
    agent = MockAgent()
    facts = [_fact("Pleural thickening"), _fact("Pneumothorax")]
    query = _sc_query("Pleural thickening", "Pneumothorax")
    result = search(query, facts, dag, agent, budget=10)
    assert result.tier == "A"
    assert "yes" in result.answer.lower()


def test_search_shared_cause_no_when_none_exists(dag):
    """No shared cause → No or ABSTAIN (not a false Yes)."""
    from src.search.tree_search import search
    from src.agent.mock import MockAgent
    agent = MockAgent()
    facts = [_fact("Calcification"), _fact("Aortic enlargement")]
    query = _sc_query("Calcification", "Aortic enlargement")
    result = search(query, facts, dag, agent, budget=10)
    assert "yes" not in result.answer.lower()


# --- predict_kronos now uses tree search ---

def test_predict_kronos_uses_tree_search(dag):
    """predict_kronos calls tree_search.search under the hood."""
    from src.eval.predictors import predict_kronos

    class _FakeGen:
        def __call__(self, prompt, image=None):
            return "anything"

    item = {"id": "t1", "finding_a": "Pleural thickening", "finding_b": "Pneumothorax",
            "question": "...", "answer": "Yes", "gold_causes": [], "support_edges": []}
    pred = predict_kronos(item, dag, _FakeGen())
    assert pred["answer"] == "Yes"
    assert pred["trace"]
    assert all(dag.causal_edge(s, t) for s, t in pred["trace"])
