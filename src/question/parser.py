import json
import re

from src.contracts import Query

NEGATION_CUES = re.compile(
    r"\b(no|not|without|clear of|rule out|free of|absent|absence|negative for)\b",
    re.IGNORECASE,
)

VALID_QTYPES = {"existential", "negation", "relational", "counting", "open", "shared_cause"}

LLM_PARSE_PROMPT = """Classify this chest X-ray question.
Return exactly one JSON object: {{"type": "<type>", "target": "<finding or null>"}}

Types: existential, negation, relational, counting, open

Examples:
Q: "Is there Cardiomegaly?"         → {{"type": "existential", "target": "Cardiomegaly"}}
Q: "Are the lungs clear?"           → {{"type": "negation", "target": null}}
Q: "Where is the Pleural effusion?" → {{"type": "relational", "target": "Pleural effusion"}}
Q: "How many findings are there?"   → {{"type": "counting", "target": null}}
Q: "What abnormality is visible?"   → {{"type": "open", "target": null}}

Q: "{question}"
"""


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
            if self.llm_client:
                llm_result = self._llm_parse(question)
                if llm_result is not None:
                    return llm_result
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

    def _llm_parse(self, question):
        """Tier-2 fallback: ask LLM to classify the question. Returns Query or None."""
        prompt = LLM_PARSE_PROMPT.format(question=question)
        raw = self.llm_client(prompt)
        if not raw:
            return None

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            return None

        qtype = data.get("type")
        if qtype not in VALID_QTYPES:
            return None

        target = data.get("target")
        if target and target not in self.finding_vocab:
            target = self._find_target(question)

        constraints = {}
        if qtype == "relational":
            if "which side" in question.lower():
                constraints = {"attr": "laterality"}
            else:
                constraints = {"attr": "location"}

        return Query(
            type=qtype,
            target=target,
            constraints=constraints,
            raw_question=question,
            parse_confidence=0.5,
            parser_tier="llm",
        )
