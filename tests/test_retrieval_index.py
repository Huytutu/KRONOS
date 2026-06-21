"""Tests for BruteForceIndex (and RagIndex if faiss is installed)."""
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
def brute_index(fixture_data):
    from src.retrieval.index import BruteForceIndex
    embeddings, cases = fixture_data
    return BruteForceIndex(embeddings, cases)


def test_search_returns_k_results(brute_index, fixture_data):
    embeddings, _ = fixture_data
    query = embeddings[0]
    results = brute_index.search(query, k=5)
    assert len(results) == 5


def test_results_sorted_descending(brute_index, fixture_data):
    embeddings, _ = fixture_data
    query = embeddings[0]
    results = brute_index.search(query, k=5)
    scores = [score for score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_self_query_ranks_first(brute_index, fixture_data):
    embeddings, cases = fixture_data
    for i in range(len(embeddings)):
        results = brute_index.search(embeddings[i], k=3)
        score, case = results[0]
        assert case["case_id"] == cases[i]["case_id"]
        assert score > 0.99


def test_k_larger_than_corpus(brute_index, fixture_data):
    embeddings, cases = fixture_data
    results = brute_index.search(embeddings[0], k=100)
    assert len(results) == len(cases)


def test_determinism(brute_index, fixture_data):
    embeddings, _ = fixture_data
    query = embeddings[3]
    first = brute_index.search(query, k=5)
    for _ in range(100):
        result = brute_index.search(query, k=5)
        assert [(s, c["case_id"]) for s, c in result] == [(s, c["case_id"]) for s, c in first]


def test_result_cases_have_expected_keys(brute_index, fixture_data):
    embeddings, _ = fixture_data
    results = brute_index.search(embeddings[0], k=3)
    for _, case in results:
        assert "case_id" in case
        assert "labels" in case
        assert "report" in case


# --- RagIndex cross-check (skipped if faiss not installed) ---

def _faiss_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _faiss_available(), reason="faiss not installed")
def test_rag_index_matches_brute_force(fixture_data):
    from src.retrieval.index import BruteForceIndex, RagIndex
    embeddings, cases = fixture_data

    brute = BruteForceIndex(embeddings, cases)
    rag = RagIndex.from_data(embeddings, cases)

    query = embeddings[2]
    brute_results = brute.search(query, k=5)
    rag_results = rag.search(query, k=5)

    for (bs, bc), (rs, rc) in zip(brute_results, rag_results):
        assert bc["case_id"] == rc["case_id"]
        assert abs(bs - rs) < 1e-5
