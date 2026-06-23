"""Tests for causal multi-hop ops — neighbors + find_path over the RGO subgraph.

Runs against the real data/ontology/causal_kg.yaml (auto-loaded as the sibling
of dag.yaml), so assertions use links verified in that graph.
"""
import pytest
from pathlib import Path
from src.ontology.dag import OntologyDAG
from src.contracts import Action
from src.tools.symbolic import run_tool

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    return OntologyDAG(
        str(DATA / "dag.yaml"),
        str(DATA / "exclusion_lists.yaml"),
        str(DATA / "anatomy_zones.yaml"),
    )


# --- loading ---

def test_causal_graph_auto_loaded(dag):
    assert dag.causal is not None
    assert dag.causal.number_of_nodes() > 0


# --- neighbors ---

def test_neighbors_caused_by_nonempty(dag):
    """A finding resolves to its RGO seed and lists its causes."""
    assert dag.causal_neighbors("Pneumothorax", "caused_by")


def test_neighbors_causes_direction(dag):
    """scleroderma may_cause cardiomegaly and interstitial lung disease."""
    effects = dag.causal_neighbors("scleroderma", "causes")
    assert "cardiomegaly" in effects
    assert "interstitial lung disease" in effects


def test_neighbors_resolves_by_label(dag):
    """A bare RGO label (lowercase) resolves as well as a finding name."""
    assert dag.causal_neighbors("cardiomegaly", "caused_by")


def test_neighbors_unknown_returns_empty(dag):
    assert dag.causal_neighbors("NotAConcept") == []


# --- find_path ---

def test_find_path_direct(dag):
    """scleroderma -> cardiomegaly is a one-hop may_cause chain."""
    path = dag.find_causal_path("scleroderma", "Cardiomegaly")
    assert path == ["scleroderma", "cardiomegaly"]


def test_find_path_unreachable_returns_none(dag):
    assert dag.find_causal_path("Aortic enlargement", "Pneumothorax") is None


def test_find_path_unknown_returns_none(dag):
    assert dag.find_causal_path("NotAConcept", "Cardiomegaly") is None


# --- tool dispatch ---

def test_neighbors_tool_dispatch(dag):
    action = Action(tool="neighbors", args={"node": "Pneumothorax", "direction": "caused_by"})
    obs = run_tool(action, [], dag, (512, 512))
    assert obs.ok
    assert isinstance(obs.result, list) and obs.result


def test_find_path_tool_dispatch(dag):
    action = Action(tool="find_path", args={"source": "scleroderma", "target": "Cardiomegaly"})
    obs = run_tool(action, [], dag, (512, 512))
    assert obs.ok
    assert obs.result == ["scleroderma", "cardiomegaly"]


def test_find_path_tool_dispatch_no_path(dag):
    action = Action(tool="find_path", args={"source": "Aortic enlargement", "target": "Pneumothorax"})
    obs = run_tool(action, [], dag, (512, 512))
    assert not obs.ok
    assert obs.result is None
