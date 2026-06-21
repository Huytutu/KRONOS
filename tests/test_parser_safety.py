"""Safety tests for QuestionParser — negation override, conservative default, determinism."""

import pytest
from src.question.parser import QuestionParser

VOCAB = {"Cardiomegaly", "Pleural effusion", "Pneumothorax"}


@pytest.fixture
def parser():
    return QuestionParser(finding_vocab=VOCAB)


class TestNegationOverride:
    """Negation cues MUST override existential classification."""

    @pytest.mark.parametrize("question", [
        "Is there no Cardiomegaly?",
        "Does this X-ray show no Pleural effusion?",
        "Rule out Pneumothorax",
        "Is there absent Cardiomegaly?",
        "Without Pleural effusion",
        "Is there any absence of Cardiomegaly?",
        "Negative for Pneumothorax",
    ])
    def test_negation_cue_wins_over_existential(self, parser, question):
        q = parser.parse(question)
        assert q.type == "negation", (
            f"Expected negation for '{question}', got '{q.type}'"
        )


class TestConservativeDefault:
    """Unrecognized questions must NEVER default to existential."""

    @pytest.mark.parametrize("question", [
        "Explain the cardiac shadow",
        "Tell me about the lungs",
        "Describe the findings",
        "What is going on here?",
        "Can you analyze this image?",
        "Is the patient healthy?",
        "Summary of radiological observations",
    ])
    def test_unknown_never_existential(self, parser, question):
        q = parser.parse(question)
        assert q.type != "existential", (
            f"Unknown question '{question}' defaulted to existential (unsafe)"
        )

    @pytest.mark.parametrize("question", [
        "Explain the cardiac shadow",
        "Tell me about the lungs",
    ])
    def test_unknown_defaults_to_relational(self, parser, question):
        q = parser.parse(question)
        assert q.type == "relational"
        assert q.parse_confidence == 0.0


class TestDeterminism:
    """Same input must always produce identical output."""

    def test_100_runs_identical(self, parser):
        question = "Is there Cardiomegaly?"
        first = parser.parse(question)
        for _ in range(100):
            result = parser.parse(question)
            assert result == first


class TestNoAnswerLeakage:
    """Query output must contain no answer-like content."""

    @pytest.mark.parametrize("question", [
        "Is there Cardiomegaly?",
        "Where is the Pleural effusion?",
        "How many findings are there?",
    ])
    def test_query_has_no_answer_field(self, parser, question):
        q = parser.parse(question)
        fields = set(q.model_fields.keys())
        answer_fields = {"answer", "result", "response", "conclusion", "diagnosis"}
        assert fields.isdisjoint(answer_fields), (
            f"Query contains answer-like fields: {fields & answer_fields}"
        )
