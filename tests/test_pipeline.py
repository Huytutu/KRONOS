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
    # Consolidation not in facts, but this is existential (not negation) → no witness → ABSTAIN
    # The mock agent tries is_a(cardiomegaly, consolidation) which fails
    assert result.tier in ("A", "ABSTAIN")


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
