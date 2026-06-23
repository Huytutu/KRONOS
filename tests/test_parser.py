"""Unit tests for QuestionParser — covers all 6 VinDr templates + negation + conservative default."""

import pytest
from src.contracts import Query
from src.question.parser import QuestionParser

VINDR_VOCAB = {
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
}


@pytest.fixture
def parser():
    return QuestionParser(finding_vocab=VINDR_VOCAB)


class TestIsThereTemplate:
    def test_does_xray_show(self, parser):
        q = parser.parse("Does this X-ray show Cardiomegaly?")
        assert q.type == "existential"
        assert q.target == "Cardiomegaly"
        assert q.constraints == {}
        assert q.parse_confidence == 1.0
        assert q.parser_tier == "rule"

    def test_is_there(self, parser):
        q = parser.parse("Is there Pleural effusion?")
        assert q.type == "existential"
        assert q.target == "Pleural effusion"


class TestWhereTemplate:
    def test_where_is(self, parser):
        q = parser.parse("Where is the Nodule/Mass?")
        assert q.type == "relational"
        assert q.target == "Nodule/Mass"
        assert q.constraints == {"attr": "location"}


class TestWhichSideTemplate:
    def test_which_side(self, parser):
        q = parser.parse("Which side shows the abnormality?")
        assert q.type == "relational"
        assert q.target is None
        assert q.constraints == {"attr": "laterality"}


class TestWhatTemplate:
    def test_what_abnormality(self, parser):
        q = parser.parse("What abnormality is visible?")
        assert q.type == "open"
        assert q.target is None
        assert q.constraints == {}


class TestHowManyTemplate:
    def test_how_many(self, parser):
        q = parser.parse("How many findings are there?")
        assert q.type == "counting"
        assert q.target is None


class TestNegation:
    def test_is_there_no(self, parser):
        q = parser.parse("Is there no Cardiomegaly?")
        assert q.type == "negation"
        assert q.target == "Cardiomegaly"

    def test_rule_out(self, parser):
        q = parser.parse("Rule out Pleural effusion")
        assert q.type == "negation"
        assert q.target == "Pleural effusion"


class TestConservativeDefault:
    def test_unknown_format_falls_to_relational(self, parser):
        q = parser.parse("Explain the cardiac shadow")
        assert q.type == "relational"
        assert q.target is None
        assert q.parse_confidence == 0.0

    def test_unknown_never_existential(self, parser):
        q = parser.parse("Tell me about the lungs")
        assert q.type != "existential"


class TestLLMFallback:
    """Tests for the tier-2 LLM fallback when rule-based parsing fails."""

    def test_valid_llm_response_produces_llm_tier(self):
        mock_llm = lambda prompt: '{"type": "existential", "target": "Cardiomegaly"}'
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Can you identify Cardiomegaly here?")
        assert q.type == "existential"
        assert q.target == "Cardiomegaly"
        assert q.parser_tier == "llm"
        assert q.parse_confidence == 0.5

    def test_llm_not_called_when_rule_matches(self):
        calls = []
        mock_llm = lambda prompt: (calls.append(1), '{"type": "open"}')[1]
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Is there Cardiomegaly?")
        assert q.parser_tier == "rule"
        assert len(calls) == 0

    def test_garbage_llm_response_keeps_rule_default(self):
        mock_llm = lambda prompt: "I don't understand the question"
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Explain the cardiac shadow")
        assert q.type == "relational"
        assert q.parse_confidence == 0.0
        assert q.parser_tier == "rule"

    def test_hallucinated_type_rejected(self):
        mock_llm = lambda prompt: '{"type": "diagnostic", "target": null}'
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Explain the cardiac shadow")
        assert q.type == "relational"
        assert q.parser_tier == "rule"

    def test_empty_llm_response_keeps_rule_default(self):
        mock_llm = lambda prompt: ""
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Explain the cardiac shadow")
        assert q.type == "relational"
        assert q.parser_tier == "rule"

    def test_hallucinated_target_falls_back_to_vocab_search(self):
        mock_llm = lambda prompt: '{"type": "existential", "target": "FakeDisease"}'
        p = QuestionParser(finding_vocab=VINDR_VOCAB, llm_client=mock_llm)
        q = p.parse("Can you see Cardiomegaly here?")
        assert q.type == "existential"
        assert q.target == "Cardiomegaly"
        assert q.parser_tier == "llm"


class TestOutputContract:
    def test_raw_question_preserved(self, parser):
        raw = "Does this X-ray show Cardiomegaly?"
        q = parser.parse(raw)
        assert q.raw_question == raw

    def test_returns_query_instance(self, parser):
        q = parser.parse("Is there Cardiomegaly?")
        assert isinstance(q, Query)

    def test_all_findings_extractable(self, parser):
        for finding in VINDR_VOCAB:
            q = parser.parse(f"Is there {finding}?")
            assert q.target == finding, f"Failed to extract target for '{finding}'"
