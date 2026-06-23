"""Tests for end-to-end pipeline — MockAgent + oracle facts."""
import pytest
from pathlib import Path
from src.contracts import PerceptualFact, Query


DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    from src.ontology.dag import OntologyDAG
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


def _oracle_facts():
    return [
        PerceptualFact(concept="Consolidation", bbox=(250, 180, 420, 350), laterality="right", conf=0.88),
        PerceptualFact(concept="Lung Opacity", bbox=(260, 350, 400, 480), laterality="right", conf=0.71),
        PerceptualFact(concept="Cardiomegaly", bbox=(100, 200, 360, 450), laterality="midline", conf=0.55),
    ]


def test_pipeline_existential(dag):
    from src.pipeline import run_with_facts
    result = run_with_facts(
        question="Is there Cardiomegaly?",
        facts=_oracle_facts(),
        dag=dag,
    )
    assert result.tier == "A"
    assert "yes" in result.answer.lower() or result.answer != ""


def test_pipeline_relational(dag):
    from src.pipeline import run_with_facts
    result = run_with_facts(
        question="Where is the Consolidation?",
        facts=_oracle_facts(),
        dag=dag,
    )
    assert result.tier == "A"
    assert result.answer != ""


def test_pipeline_counting(dag):
    from src.pipeline import run_with_facts
    result = run_with_facts(
        question="How many findings are there?",
        facts=_oracle_facts(),
        dag=dag,
    )
    assert result.tier == "A"
    assert result.answer == "3"


def test_pipeline_negation_absent(dag):
    from src.pipeline import run_with_facts
    facts = [PerceptualFact(concept="Cardiomegaly", bbox=(100, 200, 300, 400), laterality="midline", conf=0.85)]
    result = run_with_facts(
        question="Is there Consolidation?",
        facts=facts,
        dag=dag,
    )
    # Consolidation not in facts; existential (not negation). The agent checks
    # is_a(cardiomegaly, consolidation), fails, then concludes closed-world "No".
    assert result.tier == "A"
    assert result.answer == "No"


def test_pipeline_returns_search_result(dag):
    from src.pipeline import run_with_facts
    from src.contracts import SearchResult
    result = run_with_facts(
        question="How many findings are there?",
        facts=_oracle_facts(),
        dag=dag,
    )
    assert isinstance(result, SearchResult)
    assert result.tier in ("A", "B", "ABSTAIN")


class _FakeDetector:
    def detect(self, image_path):
        return _oracle_facts()


class _FakeEncoder:
    def encode(self, image):
        import numpy as np
        return np.ones(4, dtype="float32")


def test_run_sets_retriever_query_emb(dag, tmp_path):
    """run() must give the retriever the image embedding, else retrieve is inert."""
    import numpy as np
    from PIL import Image
    from src.pipeline import run
    from src.agent.mock import MockAgent
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever

    index = BruteForceIndex(np.ones((2, 4), dtype="float32"),
                            [{"case_id": "a"}, {"case_id": "b"}])
    retriever = Retriever(index, encoder=_FakeEncoder())
    assert retriever.query_emb is None

    image_path = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(image_path)

    run(str(image_path), "How many findings are there?", dag,
        _FakeDetector(), MockAgent(), retriever=retriever)

    assert retriever.query_emb is not None
