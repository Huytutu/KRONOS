"""Tests for MedGemma agent + shared prompt module. VLM inference mocked."""
from src.contracts import Action, Observation, TreeNode, PerceptualFact, Query
from src.agent.prompt import build_prompt, parse_output


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="Is there cardiac abnormality?",
        parse_confidence=1.0, parser_tier="rule",
    )


# --- Shared prompt construction (src/agent/prompt.py) ---

def test_prompt_contains_question():
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    prompt = build_prompt(node, query, k=3)
    assert "cardiac abnormality" in prompt.lower() or "cardiac_abnormality" in prompt.lower()
    assert "cardiomegaly" in prompt.lower()


def test_prompt_contains_tools():
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    prompt = build_prompt(node, query, k=3)
    assert "is_a" in prompt
    assert "inspect" in prompt


def test_prompt_contains_history():
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "pulmonary_abnormality"})
    obs = Observation(result=None, ok=False)
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[(action, obs)])
    query = _query("existential", target="cardiac_abnormality")
    prompt = build_prompt(node, query, k=3)
    assert "pulmonary_abnormality" in prompt


def test_parse_json_actions():
    raw = '[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]'
    actions = parse_output(raw)
    assert len(actions) == 1
    assert isinstance(actions[0], Action)
    assert actions[0].tool == "is_a"


def test_parse_answer():
    actions = parse_output("Answer[Yes]")
    assert len(actions) == 1
    assert actions[0] == "Yes"


def test_parse_malformed():
    actions = parse_output("I think the answer is yes but I'm not sure")
    assert actions == []


# --- MedGemma agent behavior (inference mocked) ---

def test_propose_actions_with_mock_inference():
    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=None, load_model=False)
    agent._inference_fn = lambda prompt, image: '[{"tool": "is_a", "args": {"node": "cardiomegaly", "target": "cardiac_abnormality"}}]'
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    actions = agent.propose_actions(node, query, k=3)
    assert len(actions) >= 1
    assert isinstance(actions[0], Action)


def test_propose_actions_inference_fails():
    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=None, load_model=False)
    agent._inference_fn = lambda prompt, image: "garbage output!!!"
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    actions = agent.propose_actions(node, query, k=3)
    assert actions == []


def test_propose_actions_no_model_no_inference():
    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=None, load_model=False)
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("existential", target="cardiac_abnormality")
    assert agent.propose_actions(node, query, k=3) == []


def test_agent_protocol():
    from src.agent.base import Agent
    from src.agent.medgemma import MedGemmaAgent
    agent = MedGemmaAgent(model_path=None, load_model=False)
    assert isinstance(agent, Agent)
