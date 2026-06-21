"""Tests for retrieve wiring through dispatch and search."""
import json
import numpy as np
import pytest
from pathlib import Path

from src.contracts import Action, Observation

FIXTURE = Path(__file__).parent / "fixtures" / "rag_fixture.npz"


@pytest.fixture
def retriever():
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever
    data = np.load(str(FIXTURE), allow_pickle=True)
    embeddings = data["embeddings"]
    cases = json.loads(str(data["cases"]))
    index = BruteForceIndex(embeddings, cases)
    r = Retriever(index)
    r.set_query_emb(embeddings[0])
    return r


def test_dispatch_routes_retrieve(retriever):
    from src.tools.dispatch import run_tool
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_tool(action, [], None, None, retriever=retriever)
    assert obs.ok is True
    assert len(obs.result) == 3


def test_dispatch_retrieve_no_retriever():
    from src.tools.dispatch import run_tool
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_tool(action, [], None, None, retriever=None)
    assert obs.ok is False
    assert obs.result is None


def test_dispatch_retrieve_default_no_retriever():
    """retriever param defaults to None — graceful degradation."""
    from src.tools.dispatch import run_tool
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_tool(action, [], None, None)
    assert obs.ok is False
