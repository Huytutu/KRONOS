"""Tests for LLaVA-Med agent — prompt construction + output parsing. VLM mocked."""
import pytest
from src.contracts import Action, Observation, TreeNode, PerceptualFact, Query


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="Is there cardiac abnormality?",
        parse_confidence=1.0, parser_tier="rule",
    )


def test_prompt_contains_question():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    facts = [_fact("Cardiomegaly")]
    node = TreeNode(state_facts=facts, history=[])
    query = _query("existential", target="cardiac_abnormality")
    prompt = agent.build_prompt(node, query, k=3)
    assert "cardiac abnormality" in prompt.lower() or "cardiac_abnormality" in prompt.lower()
    assert "cardiomegaly" in prompt.lower()


def test_prompt_contains_tools():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    prompt = agent.build_prompt(node, query, k=3)
    assert "is_a" in prompt
    assert "inspect" in prompt


def test_prompt_contains_history():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "pulmonary_abnormality"})
    obs = Observation(result=None, ok=False)
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[(action, obs)],
    )
    query = _query("existential", target="cardiac_abnormality")
    prompt = agent.build_prompt(node, query, k=3)
    assert "pulmonary_abnormality" in prompt


def test_parse_json_actions():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    raw = '[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]'
    actions = agent.parse_output(raw)
    assert len(actions) == 1
    assert isinstance(actions[0], Action)
    assert actions[0].tool == "is_a"


def test_parse_answer():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    raw = "Answer[Yes]"
    actions = agent.parse_output(raw)
    assert len(actions) == 1
    assert actions[0] == "Yes"


def test_parse_malformed():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    raw = "I think the answer is yes but I'm not sure"
    actions = agent.parse_output(raw)
    assert actions == []


def test_propose_actions_with_mock_inference():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    agent._inference_fn = lambda prompt, image: '[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]'
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    actions = agent.propose_actions(node, query, k=3)
    assert len(actions) >= 1
    assert isinstance(actions[0], Action)


def test_propose_actions_inference_fails():
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    agent._inference_fn = lambda prompt, image: "garbage output!!!"
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    actions = agent.propose_actions(node, query, k=3)
    assert actions == []


def test_agent_protocol():
    from src.agent.base import Agent
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=None, load_model=False)
    assert isinstance(agent, Agent)
