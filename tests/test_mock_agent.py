"""Tests for Agent Protocol + MockAgent."""
import pytest
from src.contracts import Action, TreeNode, PerceptualFact, Query


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="test", parse_confidence=1.0, parser_tier="rule",
    )


def test_mock_agent_existential():
    from src.agent.mock import MockAgent
    agent = MockAgent()
    facts = [_fact("Cardiomegaly")]
    node = TreeNode(state_facts=facts, history=[])
    query = _query("existential", target="cardiac_abnormality")
    actions = agent.propose_actions(node, query, k=3)
    assert len(actions) >= 1
    assert all(isinstance(a, (Action, str)) for a in actions)
    tool_names = [a.tool for a in actions if isinstance(a, Action)]
    assert "is_a" in tool_names


def test_mock_agent_negation():
    from src.agent.mock import MockAgent
    agent = MockAgent()
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("negation", target="Consolidation")
    actions = agent.propose_actions(node, query, k=3)
    tool_names = [a.tool for a in actions if isinstance(a, Action)]
    assert "get_exclusion_list" in tool_names


def test_mock_agent_relational():
    from src.agent.mock import MockAgent
    agent = MockAgent()
    fact = _fact("Consolidation", bbox=(250, 180, 420, 350), lat="right")
    node = TreeNode(state_facts=[fact], history=[])
    query = _query("relational", target="Consolidation", constraints={"attr": "location"})
    actions = agent.propose_actions(node, query, k=3)
    tool_names = [a.tool for a in actions if isinstance(a, Action)]
    assert "anatomy_of" in tool_names or "compose_laterality" in tool_names


def test_mock_agent_counting():
    from src.agent.mock import MockAgent
    agent = MockAgent()
    facts = [_fact("Cardiomegaly"), _fact("Consolidation")]
    node = TreeNode(state_facts=facts, history=[])
    query = _query("counting")
    actions = agent.propose_actions(node, query, k=3)
    # Counting can return an answer directly
    assert len(actions) >= 1


def test_mock_agent_returns_answer_string():
    from src.agent.mock import MockAgent
    agent = MockAgent()
    facts = [_fact("Cardiomegaly"), _fact("Consolidation")]
    node = TreeNode(state_facts=facts, history=[])
    query = _query("counting")
    actions = agent.propose_actions(node, query, k=3)
    answers = [a for a in actions if isinstance(a, str)]
    assert len(answers) >= 1


def test_mock_agent_protocol():
    from src.agent.base import Agent
    from src.agent.mock import MockAgent
    agent = MockAgent()
    assert isinstance(agent, Agent)
