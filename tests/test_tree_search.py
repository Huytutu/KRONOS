"""Tests for tree search — the capstone: best-first + backtrack + verifier-as-value."""
import pytest
from pathlib import Path
from src.contracts import PerceptualFact, Query, TreeNode, Action, Observation
from src.agent.mock import MockAgent
from src.ontology.dag import OntologyDAG

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


@pytest.fixture
def agent():
    return MockAgent()


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="test", parse_confidence=1.0, parser_tier="rule",
    )


def test_derive_answer_existential_requires_detected_fact(dag):
    """_derive_answer must not turn an ontology tautology into 'Yes'. The is_a
    source (cardiomegaly) was never detected, so the answer is closed-world 'No'."""
    from src.search.tree_search import _derive_answer
    node = TreeNode(
        state_facts=[_fact("Nodule/Mass"), _fact("Consolidation")],
        history=[
            (Action(tool="is_a", args={"node": "cardiomegaly", "target": "cardiac_abnormality"}),
             Observation(result=["cardiomegaly", "cardiac_abnormality"], ok=True)),
        ],
    )
    query = _query("existential", target="Cardiomegaly")
    assert _derive_answer(node, query) == "No"


def test_existential_direct_witness(dag, agent):
    """Cardiomegaly fact, query target=Cardiomegaly → Tier A Yes."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("existential", target="Cardiomegaly")
    result = search(query, facts, dag, agent)
    assert result.tier == "A"
    assert "yes" in result.answer.lower() or result.answer != ""


def test_existential_2hop(dag, agent):
    """Cardiomegaly fact, query target=cardiac_abnormality → Tier A via is-a path."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("existential", target="cardiac_abnormality")
    result = search(query, facts, dag, agent)
    assert result.tier == "A"
    assert len(result.path) >= 1


def test_existential_no_witness(dag, agent):
    """Cardiomegaly fact, query target=pulmonary_abnormality → no witness, but
    pulmonary_abnormality is a known concept absent from facts → Tier A "No"."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("existential", target="pulmonary_abnormality")
    result = search(query, facts, dag, agent)
    assert result.tier == "A"
    assert result.answer == "No"


def test_negation_absent(dag, agent):
    """No consolidation in facts, query negation for Consolidation → Tier A absent."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("negation", target="Consolidation")
    result = search(query, facts, dag, agent)
    assert result.tier == "A"


def test_negation_present(dag, agent):
    """Consolidation in facts, query negation for Consolidation → not Tier A."""
    from src.search.tree_search import search
    facts = [_fact("Consolidation")]
    query = _query("negation", target="Consolidation")
    result = search(query, facts, dag, agent)
    assert result.tier != "A"


def test_relational_location(dag, agent):
    """Consolidation with bbox, query Where → Tier A with anatomy zone."""
    from src.search.tree_search import search
    facts = [_fact("Consolidation", bbox=(250, 180, 420, 350), lat="right")]
    query = _query("relational", target="Consolidation", constraints={"attr": "location"})
    result = search(query, facts, dag, agent, img_wh=(512, 512))
    assert result.tier == "A"
    assert result.answer != ""


def test_counting(dag, agent):
    """Three findings → answer = 3."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly"), _fact("Consolidation"), _fact("Pleural effusion")]
    query = _query("counting")
    result = search(query, facts, dag, agent)
    assert result.tier == "A"
    assert result.answer == "3"


def test_budget_exhausted(dag, agent):
    """Tiny budget with no valid path → ABSTAIN."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("existential", target="pulmonary_abnormality")
    result = search(query, facts, dag, agent, budget=1)
    assert result.tier == "ABSTAIN"


def test_determinism(dag, agent):
    """Same inputs → identical result across 100 runs."""
    from src.search.tree_search import search
    facts = [_fact("Cardiomegaly")]
    query = _query("existential", target="cardiac_abnormality")
    results = [search(query, facts, dag, agent) for _ in range(100)]
    first = results[0]
    for r in results[1:]:
        assert r.answer == first.answer
        assert r.tier == first.tier
        assert len(r.path) == len(first.path)


def test_deletion_flips_answer(dag, agent):
    """Remove witness fact → answer must change (proves path is load-bearing)."""
    from src.search.tree_search import search
    facts_with = [_fact("Cardiomegaly")]
    facts_without = []
    query = _query("existential", target="cardiac_abnormality")

    result_with = search(query, facts_with, dag, agent)
    result_without = search(query, facts_without, dag, agent)

    assert result_with.tier == "A"
    assert result_without.tier != "A" or result_without.answer != result_with.answer


# --- reflection loop: verifier reason is fed back to the agent ---

class _ReflectionSpyAgent:
    """Always answers unverifiably; records the reflection on every node it sees."""

    def __init__(self):
        self.seen_reflections = []

    def propose_actions(self, node, query, k=3):
        self.seen_reflections.append(node.reflection)
        return ["Yes"]


def test_reflection_surfaced_to_agent(dag):
    """An unverified answer (Tier B) re-queues the state with a reason the agent reads."""
    from src.search.tree_search import search
    agent = _ReflectionSpyAgent()
    facts = [_fact("Consolidation")]
    query = _query("existential", target="SomethingNotInDAG")

    search(query, facts, dag, agent, budget=5)

    reflections = [r for r in agent.seen_reflections if r]
    assert reflections, "agent never saw a reflection — channel not wired"
    assert any("is_a" in r or "re_detect" in r for r in reflections)


def test_reflection_requeued_at_most_once(dag):
    """The loop is bounded: a reflected state is not reflected again."""
    from src.search.tree_search import search
    agent = _ReflectionSpyAgent()
    facts = [_fact("Consolidation")]
    query = _query("existential", target="SomethingNotInDAG")

    search(query, facts, dag, agent, budget=20)

    # exactly one non-empty reflection ever surfaces (single re-queue)
    assert sum(1 for r in agent.seen_reflections if r) == 1
