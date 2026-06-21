"""Tests for symbolic tool layer — thin wrappers around dag.py."""
import pytest
from pathlib import Path
from src.contracts import Action, Observation, PerceptualFact
from src.ontology.dag import OntologyDAG

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"
DAG_PATH = DATA / "dag.yaml"
EXCL_PATH = DATA / "exclusion_lists.yaml"
ZONES_PATH = DATA / "anatomy_zones.yaml"


@pytest.fixture
def dag():
    return OntologyDAG(str(DAG_PATH), str(EXCL_PATH), str(ZONES_PATH))


@pytest.fixture
def img_wh():
    return (2304, 2880)


def test_is_a_found(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "cardiac_abnormality"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert isinstance(obs.result, list)
    assert obs.result[0] == "cardiomegaly"
    assert obs.result[-1] == "cardiac_abnormality"


def test_is_a_not_found(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "pulmonary_abnormality"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is False
    assert obs.result is None


def test_disjoint_true(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="disjoint", args={"a": "pneumothorax", "b": "pleural_effusion"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert obs.result is True


def test_disjoint_false(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="disjoint", args={"a": "cardiomegaly", "b": "consolidation"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert obs.result is False


def test_anatomy_of(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="anatomy_of", args={"bbox": [100, 200, 360, 450]})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert isinstance(obs.result, str)


def test_compose_laterality(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="compose_laterality", args={"bbox": [100, 200, 360, 450]})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert obs.result in ("left", "right", "bilateral", "midline")


def test_get_exclusion_list_exists(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="get_exclusion_list", args={"name": "Cardiomegaly"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is True
    assert isinstance(obs.result, list)


def test_get_exclusion_list_missing(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="get_exclusion_list", args={"name": "NonexistentFinding"})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is False
    assert obs.result is None


def test_retrieve_not_implemented(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="retrieve", args={"image_emb": [], "k": 3})
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is False


def test_unknown_tool(dag, img_wh):
    from src.tools.symbolic import run_tool
    action = Action(tool="inspect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    obs = run_tool(action, [], dag, img_wh)
    assert obs.ok is False
