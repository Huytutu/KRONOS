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
    """Think-on-Graph over the causal KG. Beam-search backward from finding_a along
    caused_by edges; the LLM ranks which paths to explore deeper (exploration only),
    while the KG verifier alone decides the answer — a path is accepted iff its head
    also causes finding_b. The path through the KG is the trace, so every Yes is
    grounded by construction and the LLM can never inject a fabricated edge.
    Ablations: max_depth=1 (single-hop only), prune=False (no LLM pruning)."""
    a, b = item["finding_a"], item["finding_b"]
    beam = [[a]]
    visited = {a.lower()}

    for depth in range(max_depth):
        candidates = _expand(beam, dag, visited)
        if not candidates:
            break
        # Verifier decides: a path whose head also causes finding_b is a shared cause.
        hit = _first_shared_cause(candidates, b, dag)
        if hit:
            return _verified_chain(hit, b)
        # LLM explores: keep the most plausible paths to expand at the next hop.
        if depth < max_depth - 1:
            beam = _explore(candidates, a, b, gen, image, beam_width, prune)

    return _no()


def _expand(beam, dag, visited):
    """One hop deeper: extend each path by the (unvisited) causes of its head."""
    candidates = []
    for path in beam:
        for cause in dag.causal_neighbors(path[-1], "caused_by"):
            if cause.lower() not in visited:
                visited.add(cause.lower())
                candidates.append(path + [cause])
    return candidates


def _first_shared_cause(candidates, b, dag):
    """The verifier: first path whose head also causes finding_b, or None."""
    for path in candidates:
        if dag.causal_edge(path[-1], b):
            return path
    return None


def _explore(candidates, a, b, gen, image, n, prune):
    """The LLM ranks candidate paths and keeps the top-n to expand next. Pure
    exploration — it never decides the answer. prune=False skips the LLM."""
    if not prune:
        return candidates[:n]
    heads = sorted({path[-1] for path in candidates})
    prompt = (f"Chest X-ray shows {a} and {b}. Which of these conditions most likely "
              f"explains both? Options: {', '.join(heads)}. List the top {n} by name.")
    text = (gen(prompt, image) or "").lower()
    ranked = [p for p in candidates if p[-1].lower() in text]
    return (ranked or candidates)[:n]


def _verified_chain(path, b):
    """Build the KG trace from a path [finding_a, c1, ..., pivot]: consecutive
    (x, y) means y causes x, so emit y->x toward finding_a, then pivot -> finding_b.
    The pivot (shared cause) is the named answer."""
    trace = [[path[i + 1], path[i]] for i in range(len(path) - 1)]
    pivot = path[-1]
    trace.append([pivot, b])
    return {"answer": "Yes", "cause": pivot, "trace": trace}


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
