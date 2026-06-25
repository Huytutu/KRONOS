"""Tests for verifier — closure_progress + verify."""
import pytest
from pathlib import Path
from src.contracts import (
    Action, Observation, TreeNode, SearchResult,
    PerceptualFact, Query,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    from src.ontology.dag import OntologyDAG
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


def _query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="test", parse_confidence=1.0, parser_tier="rule",
    )


def _action(tool, **kwargs):
    return Action(tool=tool, args=kwargs)


def _obs(result, ok=True):
    return Observation(result=result, ok=ok)


# --- closure_progress ---

def test_progress_existential_witness_found(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[
            (_action("is_a", node="cardiomegaly", target="cardiac_abnormality"),
             _obs(["cardiomegaly", "cardiac_abnormality"])),
        ],
    )
    query = _query("existential", target="cardiac_abnormality")
    assert closure_progress(node, query, dag) == 1.0


def test_progress_existential_no_witness(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[
            (_action("is_a", node="cardiomegaly", target="pulmonary_abnormality"),
             _obs(None, ok=False)),
        ],
    )
    query = _query("existential", target="pulmonary_abnormality")
    p = closure_progress(node, query, dag)
    assert 0.0 < p < 1.0


def test_progress_negation_all_absent(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[],
        history=[
            (_action("get_exclusion_list", name="Cardiomegaly"),
             _obs(["cardiomegaly"])),
        ],
    )
    query = _query("negation", target="Cardiomegaly")
    p = closure_progress(node, query, dag)
    assert p > 0.0


def test_progress_negation_missing_list(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[],
        history=[
            (_action("get_exclusion_list", name="NonexistentFinding"),
             _obs(None, ok=False)),
        ],
    )
    query = _query("negation", target="NonexistentFinding")
    p = closure_progress(node, query, dag)
    assert p == 0.0


def test_progress_relational_resolved(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly", bbox=(100, 200, 360, 450))],
        history=[
            (_action("anatomy_of", bbox=[100, 200, 360, 450]),
             _obs("mediastinum")),
        ],
    )
    query = _query("relational", target="Cardiomegaly", constraints={"attr": "location"})
    assert closure_progress(node, query, dag) == 1.0


def test_progress_existential_has_gradient(dag):
    """Checking more facts scores strictly higher, so best-first can rank
    competing branches (the value is not a flat 0.2)."""
    from src.engine.verifier import closure_progress
    facts = [_fact("Nodule/Mass"), _fact("Consolidation")]
    query = _query("existential", target="Cardiomegaly")

    none_checked = TreeNode(state_facts=facts, history=[])
    one_checked = TreeNode(state_facts=facts, history=[
        (_action("is_a", node="nodule_mass", target="cardiomegaly"), _obs(None, ok=False)),
    ])
    both_checked = TreeNode(state_facts=facts, history=[
        (_action("is_a", node="nodule_mass", target="cardiomegaly"), _obs(None, ok=False)),
        (_action("is_a", node="consolidation", target="cardiomegaly"), _obs(None, ok=False)),
    ])
    p0 = closure_progress(none_checked, query, dag)
    p1 = closure_progress(one_checked, query, dag)
    p2 = closure_progress(both_checked, query, dag)
    assert p0 < p1 < p2 < 1.0


def test_progress_negation_has_gradient(dag):
    """After the exclusion list is fetched, checking more items scores higher."""
    from src.engine.verifier import closure_progress
    query = _query("negation", target="Consolidation")
    excl = ["consolidation", "infiltration", "lung_opacity"]
    fetched = (_action("get_exclusion_list", name="Consolidation"), _obs(excl))

    none_checked = TreeNode(state_facts=[], history=[fetched])
    one_checked = TreeNode(state_facts=[], history=[
        fetched,
        (_action("is_a", node="consolidation", target="consolidation"), _obs(None, ok=False)),
    ])
    p0 = closure_progress(none_checked, query, dag)
    p1 = closure_progress(one_checked, query, dag)
    assert p0 < p1


def test_progress_counting(dag):
    from src.engine.verifier import closure_progress
    facts = [_fact("Cardiomegaly"), _fact("Consolidation"), _fact("Pleural effusion")]
    node = TreeNode(state_facts=facts, history=[])
    query = _query("counting")
    assert closure_progress(node, query, dag) == 1.0


def test_progress_disjoint_penalty(dag):
    from src.engine.verifier import closure_progress
    node = TreeNode(
        state_facts=[_fact("Pneumothorax"), _fact("Pleural effusion")],
        history=[
            (_action("disjoint", a="pneumothorax", b="pleural_effusion"),
             _obs(True)),
        ],
    )
    query = _query("existential", target="pneumothorax")
    assert closure_progress(node, query, dag) == 0.0


# --- verify ---

def test_verify_existential_witness_must_be_a_detected_fact(dag):
    """Reproduce the witness bug. An is_a on a concept that was NOT detected
    (a DAG tautology: cardiomegaly is-a cardiac_abnormality is always true) must
    not count as evidence. Detected facts are Nodule/Mass + Consolidation, so the
    answer is closed-world 'No', not 'Yes'."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Nodule/Mass"), _fact("Consolidation")],
        history=[
            (_action("is_a", node="cardiomegaly", target="cardiac_abnormality"),
             _obs(["cardiomegaly", "cardiac_abnormality"])),
        ],
    )
    query = _query("existential", target="Cardiomegaly")
    result = verify(node, query, dag)
    assert result.tier == "A"
    assert result.answer == "No"


def test_verify_existential_tier_a(dag):
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[
            (_action("is_a", node="cardiomegaly", target="cardiac_abnormality"),
             _obs(["cardiomegaly", "cardiac_abnormality"])),
        ],
        answer="Yes",
    )
    query = _query("existential", target="cardiac_abnormality")
    result = verify(node, query, dag)
    assert result.tier == "A"
    assert result.answer == "Yes"
    assert len(result.path) == 1


def test_verify_negation_absent_tier_a(dag):
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[],
        history=[
            (_action("get_exclusion_list", name="Cardiomegaly"),
             _obs(["cardiomegaly"])),
        ],
        answer="No Cardiomegaly found",
    )
    query = _query("negation", target="Cardiomegaly")
    result = verify(node, query, dag)
    assert result.tier == "A"


def test_verify_negation_missing_list_with_answer_tier_b(dag):
    """Agent answered but exclusion list failed → tier B (not ABSTAIN)."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[],
        history=[
            (_action("get_exclusion_list", name="FakeFinding"),
             _obs(None, ok=False)),
        ],
        answer="No FakeFinding",
    )
    query = _query("negation", target="FakeFinding")
    result = verify(node, query, dag)
    assert result.tier == "B"
    assert result.answer == "No FakeFinding"


def test_verify_negation_missing_list_no_answer_abstain(dag):
    """No exclusion list AND no answer → still ABSTAIN."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[],
        history=[
            (_action("get_exclusion_list", name="FakeFinding"),
             _obs(None, ok=False)),
        ],
    )
    query = _query("negation", target="FakeFinding")
    result = verify(node, query, dag)
    assert result.tier == "ABSTAIN"


def test_verify_counting_tier_a(dag):
    from src.engine.verifier import verify
    facts = [_fact("Cardiomegaly"), _fact("Consolidation")]
    node = TreeNode(state_facts=facts, history=[], answer="2")
    query = _query("counting")
    result = verify(node, query, dag)
    assert result.tier == "A"
    assert result.answer == "2"


def test_verify_open_tier_b(dag):
    from src.engine.verifier import verify
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[], answer="Cardiomegaly visible")
    query = _query("open")
    result = verify(node, query, dag)
    assert result.tier == "B"


# --- tier B fallback: agent answer preserved when DAG can't verify ---

def test_verify_existential_answer_no_witness_tier_b(dag):
    """Agent answered 'Yes' but no is_a witness → tier B (not ABSTAIN)."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Consolidation")],
        history=[],
        answer="Yes",
    )
    query = _query("existential", target="SomethingNotInDAG")
    result = verify(node, query, dag)
    assert result.tier == "B"
    assert result.answer == "Yes"
    assert result.conf == 0.3


def test_verify_existential_no_answer_still_abstain(dag):
    """No witness AND no answer → ABSTAIN."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Consolidation")],
        history=[],
    )
    query = _query("existential", target="SomethingNotInDAG")
    result = verify(node, query, dag)
    assert result.tier == "ABSTAIN"


def test_verify_relational_answer_no_tool_tier_b(dag):
    """Agent answered 'left lung' but no anatomy_of in history → tier B."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[],
        answer="left lung",
    )
    query = _query("relational", target="Cardiomegaly", constraints={"attr": "location"})
    result = verify(node, query, dag)
    assert result.tier == "B"
    assert result.answer == "left lung"
    assert result.conf == 0.3


def test_verify_relational_no_answer_still_abstain(dag):
    """No anatomy tool AND no answer → ABSTAIN."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[],
    )
    query = _query("relational", target="Cardiomegaly", constraints={"attr": "location"})
    result = verify(node, query, dag)
    assert result.tier == "ABSTAIN"


def test_tier_a_unchanged_with_witness(dag):
    """Tier A still works — existential with witness is A, not B."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly")],
        history=[
            (_action("is_a", node="cardiomegaly", target="cardiac_abnormality"),
             _obs(["cardiomegaly", "cardiac_abnormality"])),
        ],
        answer="Yes",
    )
    query = _query("existential", target="cardiac_abnormality")
    result = verify(node, query, dag)
    assert result.tier == "A"


def test_tier_a_unchanged_relational_with_tool(dag):
    """Tier A still works — relational with anatomy_of is A, not B."""
    from src.engine.verifier import verify
    node = TreeNode(
        state_facts=[_fact("Cardiomegaly", bbox=(100, 200, 360, 450))],
        history=[
            (_action("anatomy_of", bbox=[100, 200, 360, 450]),
             _obs("mediastinum")),
        ],
        answer="mediastinum",
    )
    query = _query("relational", target="Cardiomegaly", constraints={"attr": "location"})
    result = verify(node, query, dag)
    assert result.tier == "A"


# --- explain: the reflection reason fed back to the agent ---

def test_explain_existential_points_to_next_tool(dag):
    from src.engine.verifier import explain
    node = TreeNode(state_facts=[_fact("Consolidation")], history=[])
    query = _query("existential", target="SomethingNotInDAG")
    reason = explain(node, query, dag)
    assert reason
    assert any(t in reason for t in ("is_a", "re_detect", "neighbors", "find_path"))


def test_explain_negation_points_to_exclusion_list(dag):
    from src.engine.verifier import explain
    node = TreeNode(state_facts=[], history=[])
    query = _query("negation", target="Cardiomegaly")
    reason = explain(node, query, dag)
    assert reason
    assert "get_exclusion_list" in reason


def test_explain_relational_points_to_localization_tool(dag):
    from src.engine.verifier import explain
    node = TreeNode(state_facts=[_fact("Cardiomegaly")], history=[])
    query = _query("relational", target="Cardiomegaly", constraints={"attr": "location"})
    reason = explain(node, query, dag)
    assert reason
    assert any(t in reason for t in ("anatomy_of", "compose_laterality"))
