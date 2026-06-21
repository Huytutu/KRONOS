"""Tests for Retriever and run_retrieve tool wrapper."""
import json
import numpy as np
import pytest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "rag_fixture.npz"


@pytest.fixture
def fixture_data():
    data = np.load(str(FIXTURE), allow_pickle=True)
    embeddings = data["embeddings"]
    cases = json.loads(str(data["cases"]))
    return embeddings, cases


@pytest.fixture
def retriever(fixture_data):
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever
    embeddings, cases = fixture_data
    index = BruteForceIndex(embeddings, cases)
    r = Retriever(index)
    r.set_query_emb(embeddings[0])
    return r


# --- Retriever tests ---

def test_retrieve_returns_k_cases(retriever):
    results = retriever.retrieve(k=3)
    assert len(results) == 3


def test_retrieve_default_k(retriever):
    results = retriever.retrieve()
    assert len(results) == 5


def test_retrieve_without_query_emb(fixture_data):
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever
    embeddings, cases = fixture_data
    index = BruteForceIndex(embeddings, cases)
    r = Retriever(index)
    # no set_query_emb called
    results = r.retrieve(k=3)
    assert results == []


def test_retrieve_cases_have_score(retriever):
    results = retriever.retrieve(k=3)
    for case in results:
        assert "score" in case
        assert "case_id" in case


# --- run_retrieve tool wrapper tests ---

def test_run_retrieve_ok(retriever):
    from src.contracts import Action
    from src.retrieval.tool import run_retrieve
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_retrieve(action, retriever)
    assert obs.ok is True
    assert len(obs.result) == 3


def test_run_retrieve_no_retriever():
    from src.contracts import Action
    from src.retrieval.tool import run_retrieve
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_retrieve(action, None)
    assert obs.ok is False
    assert obs.result is None


def test_run_retrieve_no_query_emb(fixture_data):
    from src.contracts import Action
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever
    from src.retrieval.tool import run_retrieve
    embeddings, cases = fixture_data
    index = BruteForceIndex(embeddings, cases)
    r = Retriever(index)
    action = Action(tool="retrieve", args={"k": 3})
    obs = run_retrieve(action, r)
    assert obs.ok is False


def test_run_retrieve_default_k(retriever):
    from src.contracts import Action
    from src.retrieval.tool import run_retrieve
    action = Action(tool="retrieve", args={})
    obs = run_retrieve(action, retriever)
    assert obs.ok is True
    assert len(obs.result) == 5


def test_run_retrieve_honours_k(retriever):
    from src.contracts import Action
    from src.retrieval.tool import run_retrieve
    action = Action(tool="retrieve", args={"k": 2})
    obs = run_retrieve(action, retriever)
    assert obs.ok is True
    assert len(obs.result) == 2
