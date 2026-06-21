"""Tests for E_perc proposer — direct name-match on perception facts."""

import pytest
from src.contracts import Query, PerceptualFact, Candidate
from src.proposers.perc import EPerc


def make_query(qtype, target=None, constraints=None):
    return Query(
        type=qtype, target=target, constraints=constraints or {},
        raw_question="test", parse_confidence=1.0, parser_tier="rule",
    )


FACTS = [
    PerceptualFact(concept="Cardiomegaly", bbox=(170, 280, 400, 420), laterality="midline", conf=0.85),
    PerceptualFact(concept="Consolidation", bbox=(50, 100, 200, 250), laterality="right", conf=0.72),
    PerceptualFact(concept="Pleural effusion", bbox=(300, 100, 480, 400), laterality="left", conf=0.60),
]


@pytest.fixture
def eperc():
    return EPerc()


# --- existential ---

class TestExistential:
    def test_target_present(self, eperc):
        q = make_query("existential", target="Cardiomegaly")
        c = eperc.propose(q, FACTS)
        assert c is not None
        assert c.answer == "Yes"
        assert c.head_id == "E_perc"
        assert c.conf == 0.85

    def test_target_absent(self, eperc):
        q = make_query("existential", target="Pneumothorax")
        c = eperc.propose(q, FACTS)
        assert c is not None
        assert c.answer == "No"

    def test_open_scope(self, eperc):
        q = make_query("existential", target=None, constraints={"scope": "open"})
        c = eperc.propose(q, FACTS)
        assert c is not None
        assert "Cardiomegaly" in c.answer
        assert "Consolidation" in c.answer
        assert "Pleural effusion" in c.answer

    def test_empty_facts(self, eperc):
        q = make_query("existential", target="Cardiomegaly")
        c = eperc.propose(q, [])
        assert c.answer == "No"


# --- negation ---

class TestNegation:
    def test_target_found(self, eperc):
        q = make_query("negation", target="Cardiomegaly")
        c = eperc.propose(q, FACTS)
        assert "present" in c.answer.lower() or "found" in c.answer.lower()

    def test_target_not_found(self, eperc):
        q = make_query("negation", target="Pneumothorax")
        c = eperc.propose(q, FACTS)
        assert "no" in c.answer.lower()

    def test_empty_facts(self, eperc):
        q = make_query("negation", target="Cardiomegaly")
        c = eperc.propose(q, [])
        assert "no" in c.answer.lower()


# --- relational ---

class TestRelational:
    def test_laterality(self, eperc):
        q = make_query("relational", target="Consolidation", constraints={"attr": "laterality"})
        c = eperc.propose(q, FACTS)
        assert c is not None
        assert c.answer == "right"

    def test_location(self, eperc):
        q = make_query("relational", target="Cardiomegaly", constraints={"attr": "location"})
        c = eperc.propose(q, FACTS)
        assert c is not None

    def test_target_not_found(self, eperc):
        q = make_query("relational", target="Pneumothorax", constraints={"attr": "laterality"})
        c = eperc.propose(q, FACTS)
        assert c is None


# --- counting ---

class TestCounting:
    def test_count(self, eperc):
        q = make_query("counting")
        c = eperc.propose(q, FACTS)
        assert c is not None
        assert c.answer == "3"

    def test_empty(self, eperc):
        q = make_query("counting")
        c = eperc.propose(q, [])
        assert c.answer == "0"


# --- output contract ---

class TestContract:
    def test_returns_candidate(self, eperc):
        q = make_query("existential", target="Cardiomegaly")
        c = eperc.propose(q, FACTS)
        assert isinstance(c, Candidate)

    def test_head_id_is_eperc(self, eperc):
        q = make_query("existential", target="Cardiomegaly")
        c = eperc.propose(q, FACTS)
        assert c.head_id == "E_perc"

    def test_conf_in_range(self, eperc):
        q = make_query("existential", target="Cardiomegaly")
        c = eperc.propose(q, FACTS)
        assert 0.0 <= c.conf <= 1.0
