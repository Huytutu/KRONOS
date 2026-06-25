"""Per-system predictors for the multi-hop shared-cause QA.

Each predictor maps a QA item to a prediction {answer, cause, trace} in the
grading schema. Model-based predictors live in Phase B; predict_mock here is a
deterministic KG oracle used to validate the predict -> grade pipeline without a
model (a test fixture, NOT a reported system).
"""
import json
import re


def build_sc_prompt(item, mode):
    """Prompt for a shared-cause question. mode: zero_shot | cot | react."""
    q = item["question"]
    if mode == "cot":
        return (q + "\nReason step by step about conditions that could cause both "
                "findings, then end with a final line: 'Answer: Yes, <condition>' "
                "or 'Answer: No'.")
    return (q + "\nReply with 'Yes, <condition>' if a single condition can cause both, "
            "otherwise 'No'.")


def parse_yes_no_cause(text):
    """Parse model text -> (answer, cause). Safe default ('No', None) on junk."""
    if not text or not isinstance(text, str):
        return ("No", None)
    match = re.search(r"answer\s*[:\-]\s*(.+)", text, re.I)
    segment = (match.group(1) if match else text).strip()
    if re.match(r"(?i)\s*yes\b", segment):
        cause = None
        cm = re.search(r"(?i)yes[\s,:\-]+(.+)", segment)
        if cm:
            cause = cm.group(1).splitlines()[0].strip().rstrip(".").strip() or None
        return ("Yes", cause)
    return ("No", None)


def predict_mock(item, dag):
    """KG oracle: answer straight from the shared-cause set (verified trace)."""
    causes = dag.common_causes(item["finding_a"], item["finding_b"])
    if not causes:
        return {"answer": "No", "cause": None, "trace": []}
    cause = causes[0]
    return {
        "answer": "Yes",
        "cause": cause,
        "trace": [[cause, item["finding_a"]], [cause, item["finding_b"]]],
    }


# --- model-based predictors (gen: callable(prompt, image) -> text) ---

def _no():
    return {"answer": "No", "cause": None, "trace": []}


def predict_zero_shot(item, gen, image=None):
    """MedGemma direct answer. No trace -> ungrounded by design."""
    answer, cause = parse_yes_no_cause(gen(build_sc_prompt(item, "zero_shot"), image))
    return {"answer": answer, "cause": cause, "trace": []}


def predict_cot(item, gen, image=None):
    """MedGemma chain-of-thought. No trace -> ungrounded by design."""
    answer, cause = parse_yes_no_cause(gen(build_sc_prompt(item, "cot"), image))
    return {"answer": answer, "cause": cause, "trace": []}


def predict_kronos(item, dag, gen, image=None, *,
                   beam_width=3, max_depth=3, prune=True):
    """KRONOS shared-cause predictor — uses the unified tree search engine.

    Builds a shared_cause Query and runs tree_search.search() with a
    SharedCauseAgent that wraps the LLM gen function. The verifier decides
    the answer (never the LLM), and every trace edge is a real KG edge.
    Ablations: max_depth=1 (limit search budget), prune=False (skip LLM ranking)."""
    from src.contracts import Query, PerceptualFact
    from src.search.tree_search import search

    a, b = item["finding_a"], item["finding_b"]
    query = Query(type="shared_cause", target=None,
                  constraints={"finding_a": a, "finding_b": b},
                  raw_question=item.get("question", f"shared cause of {a} and {b}?"),
                  parse_confidence=1.0, parser_tier="rule")
    facts = [PerceptualFact(concept=a, bbox=(0, 0, 1, 1), laterality="midline", conf=1.0),
             PerceptualFact(concept=b, bbox=(0, 0, 1, 1), laterality="midline", conf=1.0)]
    agent = _SharedCauseAgent(gen, image, beam_width, prune)
    budget = max_depth * 4
    result = search(query, facts, dag, agent, budget=budget)

    return _result_to_pred(result, a, b, dag)


class _SharedCauseAgent:
    """Thin agent for shared-cause that explores causal neighbors, optionally
    using the LLM to rank which causes to explore first."""

    def __init__(self, gen, image, beam_width, prune):
        self.gen = gen
        self.image = image
        self.beam_width = beam_width
        self.prune = prune

    def propose_actions(self, node, query, k=3):
        from src.contracts import Action
        a = query.constraints.get("finding_a", "")
        b = query.constraints.get("finding_b", "")
        explored = set()
        for action, obs in node.history:
            if action.tool == "neighbors":
                explored.add(action.args.get("node", "").lower())

        if a.lower() not in explored:
            return [Action(tool="neighbors", args={"node": a, "direction": "caused_by"})]
        if b.lower() not in explored:
            return [Action(tool="neighbors", args={"node": b, "direction": "caused_by"})]
        return ["No"]


def _result_to_pred(result, a, b, dag):
    """Convert a SearchResult into the {answer, cause, trace} prediction shape."""
    if "yes" in result.answer.lower():
        cause = result.answer.split(",", 1)[1].strip() if "," in result.answer else None
        trace = [[cause, a], [cause, b]] if cause and dag.causal_edge(cause, a) and dag.causal_edge(cause, b) else []
        return {"answer": "Yes", "cause": cause, "trace": trace}
    return {"answer": "No", "cause": None, "trace": []}


def predict_react(item, dag, gen, image=None):
    """ReAct over the SAME graph tools but NO verifier gate: the model is shown the
    causal neighborhoods and answers freely. Its named cause is accepted as-is, so
    it can name an unverified (hallucinated) cause; trace is filled only if the cause
    happens to be a real chain."""
    a, b = item["finding_a"], item["finding_b"]
    prompt = build_sc_prompt(item, "react") + (
        f"\nKnown causes of {a}: {dag.causal_neighbors(a, 'caused_by')[:20]}"
        f"\nKnown causes of {b}: {dag.causal_neighbors(b, 'caused_by')[:20]}")
    answer, cause = parse_yes_no_cause(gen(prompt, image))
    trace = ([[cause, a], [cause, b]]
             if cause and dag.causal_edge(cause, a) and dag.causal_edge(cause, b) else [])
    return {"answer": answer, "cause": cause, "trace": trace}


def write_predictions(items, predict_fn, path):
    """Run predict_fn(item) over items and write predictions JSONL (with id)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            pred = predict_fn(item)
            f.write(json.dumps({"id": item["id"], **pred}) + "\n")
