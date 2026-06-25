"""Automatic, deterministic grading for the multi-hop shared-cause QA subset.

Grades a system's predictions against KG-derived gold answers — no model needed.
See multihop_qa_SPEC.md §5. Prediction shape:
    {"id", "answer": "Yes"/"No", "cause": <disorder|null>, "trace": [[src,tgt], ...]}
"""


def grade_item(item, pred, dag):
    """Per-item flags: binary_correct, name_correct, grounded, hallucinated."""
    gold_causes = {c.lower() for c in item.get("gold_causes", [])}
    p_answer = pred.get("answer", "")
    p_cause = (pred.get("cause") or "").lower()
    trace = pred.get("trace") or []

    binary_correct = p_answer == item["answer"]
    name_correct = grounded = hallucinated = False

    if p_answer == "Yes":
        name_correct = p_cause in gold_causes
        grounded = _trace_supports(trace, item, dag)
        hallucinated = not name_correct

    return {
        "binary_correct": binary_correct,
        "name_correct": name_correct,
        "grounded": grounded,
        "hallucinated": hallucinated,
    }


def _trace_supports(trace, item, dag):
    """True if every edge in the claimed trace is a real may_cause edge and the
    trace is a connected shared-cause chain: one node (the shared cause) from
    which both finding_a and finding_b are reachable along the trace edges.
    Handles multi-hop chains, not just two edges sharing a single cause."""
    if not trace:
        return False
    for edge in trace:
        if len(edge) != 2 or not dag.causal_edge(edge[0], edge[1]):
            return False
    a = item["finding_a"].lower()
    b = item["finding_b"].lower()
    nodes = {str(n).lower() for edge in trace for n in edge}
    return any(_reaches(trace, pivot, a) and _reaches(trace, pivot, b)
              for pivot in nodes)


def _reaches(trace, src, dst):
    """True if dst is reachable from src following directed trace edges."""
    seen = {src}
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for s, t in trace:
            s, t = str(s).lower(), str(t).lower()
            if s == node and t not in seen:
                seen.add(t)
                stack.append(t)
    return False


def deletion_holds(item, dag):
    """Load-bearing check: removing the item's support_edges leaves no common cause
    (the gold 'Yes' depended on that chain, not a guess)."""
    return not dag.common_causes(item["finding_a"], item["finding_b"],
                                 removed=item.get("support_edges"))


def load_bearing_rate(items, dag):
    """Fraction of single-cause Yes items whose answer is path-dependent (deletion flips it)."""
    single = [it for it in items if it["answer"] == "Yes" and it.get("single_cause")]
    return _rate([deletion_holds(it, dag) for it in single])


def grade(items, preds, dag):
    """Aggregate the five metrics (denominators per multihop_qa_SPEC §5)."""
    by_id = {p["id"]: p for p in preds}
    flags = [(it, grade_item(it, by_id.get(it["id"], {}), dag)) for it in items]

    gold_yes = [f for it, f in flags if it["answer"] == "Yes"]
    pred_yes = [f for it, f in flags if by_id.get(it["id"], {}).get("answer") == "Yes"]

    return {
        "n": len(items),
        "binary_accuracy": _rate([f["binary_correct"] for _, f in flags]),
        "name_accuracy": _rate([f["name_correct"] for f in gold_yes]),
        "grounding_rate": _rate([f["grounded"] for f in pred_yes]),
        "hallucination_rate": _rate([f["hallucinated"] for f in pred_yes]),
        "load_bearing_rate": load_bearing_rate(items, dag),
    }


def _rate(flags):
    return sum(flags) / len(flags) if flags else 0.0
