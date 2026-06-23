import re

from src.contracts import Query

# Words that signal a question asks about ABSENCE, not presence.
# Word-boundary matching matters here, so this one stays a compiled regex.
NEGATION_CUES = re.compile(
    r"\b(no|not|without|clear of|rule out|free of|absent|absence|negative for)\b",
    re.IGNORECASE,
)


class QuestionParser:
    """Turns a question into a Query(type, target, constraints). Rule-based; LLM is a future fallback."""

    def __init__(self, finding_vocab, llm_client=None):
        self.finding_vocab = finding_vocab
        self.llm_client = llm_client

    def parse(self, question):
        q = question.lower()
        target = self._find_target(question)

        # Negation is checked first: calling an absence question "existential"
        # would let the engine stop early and answer wrong.
        if NEGATION_CUES.search(question):
            qtype, target, constraints, conf = "negation", target, {}, 1.0
        elif "how many" in q:
            qtype, target, constraints, conf = "counting", None, {}, 1.0
        elif "where is" in q:
            qtype, target, constraints, conf = "relational", target, {"attr": "location"}, 1.0
        elif "which side" in q:
            qtype, target, constraints, conf = "relational", None, {"attr": "laterality"}, 1.0
        elif "what abnormality" in q or "what finding" in q:
            # Free-text descriptive question → open (served as advisory Tier B),
            # not existential. Existential with no target can never find a
            # witness, so it would always abstain.
            qtype, target, constraints, conf = "open", None, {}, 1.0
        elif "is there" in q or ("does" in q and "show" in q):
            qtype, target, constraints, conf = "existential", target, {}, 1.0
        else:
            # Unknown wording: fall back to "relational", never "existential".
            # A wrong "existential" guess is unsafe; "relational" is the safe default.
            qtype, target, constraints, conf = "relational", target, {}, 0.0

        return Query(
            type=qtype,
            target=target,
            constraints=constraints,
            raw_question=question,
            parse_confidence=conf,
            parser_tier="rule",
        )

    def parse_batch(self, questions):
        return [self.parse(q) for q in questions]

    def _find_target(self, question):
        """Return the first finding name that appears in the question, or None."""
        q = question.lower()
        for finding in self.finding_vocab:
            if finding.lower() in q:
                return finding
        return None
