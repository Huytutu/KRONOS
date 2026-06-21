"""Faithfulness guard tests — prove retrieve is inert to Tier-A."""
import pytest
from pathlib import Path
from src.contracts import Action, Observation, TreeNode, PerceptualFact, Query
from src.engine.verifier import closure_progress, verify

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    from src.ontology.dag import OntologyDAG
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None):
    return Query(type=qtype, target=target, constraints={},
                 raw_question="test", parse_confidence=1.0, parser_tier="rule")


RETRIEVE_STEP = (
    Action(tool="retrieve", args={"k": 3}),
    Observation(result=[{"case_id": "x", "score": 0.9}], ok=True),
)

IS_A_STEP = (
    Action(tool="is_a", args={"node": "Cardiomegaly", "target": "Abnormality"}),
    Observation(result=["Cardiomegaly", "Abnormality"], ok=True),
)


# --- closure_progress identical with/without retrieve ---

def test_progress_unchanged_with_retrieve(dag):
    query = _query("existential", target="Cardiomegaly")
    facts = [_fact("Cardiomegaly")]

    without = TreeNode(state_facts=facts, history=[IS_A_STEP])
    with_ret = TreeNode(state_facts=facts, history=[RETRIEVE_STEP, IS_A_STEP])

    assert closure_progress(without, query, dag) == closure_progress(with_ret, query, dag)


def test_progress_unchanged_retrieve_only(dag):
    query = _query("existential", target="Cardiomegaly")
    facts = [_fact("Cardiomegaly")]

    empty = TreeNode(state_facts=facts, history=[])
    ret_only = TreeNode(state_facts=facts, history=[RETRIEVE_STEP])

    assert closure_progress(empty, query, dag) == closure_progress(ret_only, query, dag)


# --- deletion test ---

def test_deletion_remove_retrieve_tier_a_unchanged(dag):
    """Removing retrieve from a Tier-A trace does not change the answer."""
    query = _query("existential", target="Cardiomegaly")
    facts = [_fact("Cardiomegaly")]

    with_ret = TreeNode(state_facts=facts, history=[RETRIEVE_STEP, IS_A_STEP], answer="Yes")
    without = TreeNode(state_facts=facts, history=[IS_A_STEP], answer="Yes")

    result_with = verify(with_ret, query, dag)
    result_without = verify(without, query, dag)

    assert result_with.tier == "A"
    assert result_without.tier == "A"
    assert result_with.answer == result_without.answer


def test_deletion_remove_witness_flips_tier(dag):
    """Removing the is_a witness flips from Tier-A to ABSTAIN."""
    query = _query("existential", target="Abnormality")
    facts = [_fact("Cardiomegaly")]

    with_witness = TreeNode(state_facts=facts, history=[IS_A_STEP], answer="Yes")
    without_witness = TreeNode(state_facts=facts, history=[], answer="Yes")

    assert verify(with_witness, query, dag).tier == "A"
    assert verify(without_witness, query, dag).tier != "A"


# --- retrieve-only node never Tier-A ---

def test_retrieve_only_node_never_tier_a_existential(dag):
    """Retrieve-only node can't reach Tier-A for existential requiring is_a hop."""
    query = _query("existential", target="Abnormality")
    facts = [_fact("Cardiomegaly")]
    node = TreeNode(state_facts=facts, history=[RETRIEVE_STEP], answer="Yes")
    assert verify(node, query, dag).tier != "A"


def test_retrieve_only_node_never_tier_a_negation(dag):
    """Retrieve-only node can't reach Tier-A for negation (needs get_exclusion_list)."""
    query = _query("negation", target="Pneumothorax")
    facts = [_fact("Cardiomegaly")]
    node = TreeNode(state_facts=facts, history=[RETRIEVE_STEP], answer="No Pneumothorax found")
    assert verify(node, query, dag).tier != "A"


def test_retrieve_only_node_never_tier_a_relational(dag):
    """Retrieve-only node can't reach Tier-A for relational (needs anatomy_of/compose_laterality)."""
    query = _query("relational", target="Cardiomegaly")
    facts = [_fact("Cardiomegaly")]
    node = TreeNode(state_facts=facts, history=[RETRIEVE_STEP], answer="left lung")
    assert verify(node, query, dag).tier != "A"
