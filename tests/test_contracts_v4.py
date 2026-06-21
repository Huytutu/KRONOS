"""RED tests for v4 contract types."""
import pytest
from src.contracts import (
    Action, Observation, TreeNode, SearchResult,
    Tier, PerceptualFact, Query,
)


def test_action_symbolic():
    a = Action(tool="is_a", args={"node": "cardiomegaly", "target": "cardiac_abnormality"})
    assert a.kind == "symbolic"
    assert a.tool == "is_a"


def test_action_visual_kind():
    a = Action(tool="inspect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    assert a.kind == "visual"


def test_observation():
    obs = Observation(result=["cardiomegaly", "cardiac_abnormality"], ok=True)
    assert obs.ok is True
    assert len(obs.result) == 2


def test_tree_node_empty():
    node = TreeNode(state_facts=[], history=[])
    assert node.answer is None
    assert node.reward == 0.0
    assert node.parent_id is None
    assert node.reflection == ""


def test_tree_node_with_history():
    fact = PerceptualFact(
        concept="Cardiomegaly", bbox=(100, 200, 300, 400),
        laterality="midline", conf=0.85,
    )
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "cardiac_abnormality"})
    obs = Observation(result=["cardiomegaly", "cardiac_abnormality"], ok=True)
    node = TreeNode(
        state_facts=[fact],
        history=[(action, obs)],
        reward=1.0,
    )
    assert len(node.history) == 1
    assert node.reward == 1.0


def test_search_result_tier_a():
    action = Action(tool="is_a", args={"node": "x", "target": "y"})
    obs = Observation(result=True, ok=True)
    result = SearchResult(
        answer="Yes", tier="A",
        path=[(action, obs)], conf=0.88,
    )
    assert result.tier == "A"


def test_search_result_abstain():
    result = SearchResult(answer="", tier="ABSTAIN", path=[], conf=0.0)
    assert result.tier == "ABSTAIN"


def test_tier_values():
    for t in ("A", "B", "ABSTAIN"):
        result = SearchResult(answer="x", tier=t, path=[], conf=0.5)
        assert result.tier == t


def test_qtype_open():
    q = Query(
        type="open", target=None, constraints={},
        raw_question="What abnormality?", parse_confidence=1.0, parser_tier="rule",
    )
    assert q.type == "open"


def test_candidate_still_exists_for_backwards_compat():
    """Candidate kept temporarily for perc.py; will be removed when perc.py is retired."""
    from src.contracts import Candidate  # noqa: F401
