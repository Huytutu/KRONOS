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


def _verified(cause, a, b):
    return {"answer": "Yes", "cause": cause, "trace": [[cause, a], [cause, b]]}


def predict_zero_shot(item, gen, image=None):
    """MedGemma direct answer. No trace -> ungrounded by design."""
    answer, cause = parse_yes_no_cause(gen(build_sc_prompt(item, "zero_shot"), image))
    return {"answer": answer, "cause": cause, "trace": []}


def predict_cot(item, gen, image=None):
    """MedGemma chain-of-thought. No trace -> ungrounded by design."""
    answer, cause = parse_yes_no_cause(gen(build_sc_prompt(item, "cot"), image))
    return {"answer": answer, "cause": cause, "trace": []}


def predict_kronos(item, dag, gen, image=None, multi_hop=True, reflection=True):
    """Model-in-loop, KG-gated: MedGemma proposes a cause; it is accepted ONLY if
    the KG verifies it causes both findings. On rejection, reflect once, then fall
    back to multi-hop graph search. Every Yes is KG-verified (grounded by design).
    Ablations: multi_hop=False (no graph fallback), reflection=False (single attempt)."""
    a, b = item["finding_a"], item["finding_b"]

    _, cand = parse_yes_no_cause(gen(build_sc_prompt(item, "zero_shot"), image))
    if cand and dag.causal_edge(cand, a) and dag.causal_edge(cand, b):
        return _verified(cand, a, b)

    if reflection:
        prompt = build_sc_prompt(item, "zero_shot") + (
            f"\n(Note: '{cand}' does not cause both {a} and {b} in the reference. "
            "Propose another single condition that causes both, or answer No.)")
        _, cand2 = parse_yes_no_cause(gen(prompt, image))
        if cand2 and dag.causal_edge(cand2, a) and dag.causal_edge(cand2, b):
            return _verified(cand2, a, b)

    if multi_hop:
        for d in dag.causal_neighbors(a, "caused_by"):
            if dag.causal_edge(d, b):
                return _verified(d, a, b)

    return _no()


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
