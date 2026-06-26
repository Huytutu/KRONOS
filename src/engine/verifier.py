"""Verifier — two roles:
1. closure_progress(node, query, dag) → float  (search value, guides tree)
2. verify(node, query, dag) → SearchResult      (terminal gate, assigns tier)

Deterministic. No weights. No LLM self-eval.
"""
from src.contracts import TreeNode, Query, SearchResult


def closure_progress(node, query, dag):
    if _has_disjoint_violation(node):
        return 0.0

    qtype = query.type

    if qtype == "existential":
        return _progress_existential(node, query, dag)
    if qtype == "negation":
        return _progress_negation(node, query, dag)
    if qtype == "relational":
        return _progress_relational(node, query, dag)
    if qtype == "counting":
        return 1.0
    if qtype == "shared_cause":
        return _progress_shared_cause(node, query, dag)
    if qtype == "open":
        return _progress_open(node)
    return 0.0


def verify(node, query, dag):
    qtype = query.type
    path = list(node.history)

    if qtype == "open":
        if node.answer:
            tier = "A" if node.state_facts else "B"
            return SearchResult(answer=node.answer, tier=tier, path=path, conf=_min_conf(node))
        return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)

    if qtype == "existential":
        return _verify_existential(node, query, dag)
    if qtype == "negation":
        return _verify_negation(node, query, dag)
    if qtype == "relational":
        return _verify_relational(node, query, dag)
    if qtype == "counting":
        return _verify_counting(node, query, dag)
    if qtype == "shared_cause":
        return _verify_shared_cause(node, query, dag)

    return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)


def explain(node, query, dag):
    """Short, deterministic reason a node failed to reach Tier A — fed back to the
    agent as a reflection so it can revise on its next attempt. Mirrors the failure
    branches of verify() per query type."""
    qtype = query.type
    target = query.target or "the target"

    if qtype == "existential":
        return (f"No detected finding is-a '{target}', and '{target}' is not a known "
                f"ontology concept, so absence cannot be confirmed. Try re_detect near a "
                f"suspected region then is_a on new findings, or neighbors/find_path to "
                f"relate '{target}' causally.")

    if qtype == "negation":
        if _get_fetched_exclusion_list(node) is None:
            return (f"The exclusion list for '{target}' was not fetched. Call "
                    f"get_exclusion_list('{target}') first, then check each item with is_a.")
        return (f"A detected finding matches the exclusion list for '{target}', so it is "
                f"present — '{target}' cannot be confirmed absent.")

    if qtype == "relational":
        return ("Location is unresolved. Call anatomy_of(bbox) or compose_laterality(bbox) "
                "on the target finding's bbox.")

    return "Insufficient verified evidence; gather more before answering."


# --- existential ---

def _progress_existential(node, query, dag):
    if _has_witness(node, dag) or _has_direct_match(node, query, dag):
        return 1.0
    # Partial credit for how much of the evidence has been examined, so the
    # frontier can rank a node that checked more facts above one that checked few.
    return 0.1 + _fraction_facts_checked(node, dag) * 0.7


def _fraction_facts_checked(node, dag):
    """Fraction of detected facts that have been the source of an is_a action."""
    if not node.state_facts:
        return 0.0
    fact_slugs = {dag.resolve_slug(f.concept) for f in node.state_facts}
    checked = {
        dag.resolve_slug(action.args.get("node", ""))
        for action, obs in node.history
        if action.tool == "is_a"
    }
    return len(fact_slugs & checked) / len(fact_slugs)


def _verify_existential(node, query, dag):
    path = list(node.history)
    if _has_witness(node, dag) or _has_direct_match(node, query, dag):
        return SearchResult(
            answer=node.answer or "Yes", tier="A", path=path, conf=_min_conf(node),
        )
    # Closed-world absence: if the target is a known concept and no detected
    # finding is (or is-a) the target, "No" is a verified answer, not a guess.
    target_id = _resolve_target(query, dag)
    if target_id and not _any_fact_is_a(node, target_id, dag):
        return SearchResult(
            answer=node.answer or "No", tier="A", path=path, conf=_min_conf(node),
        )
    if node.answer:
        return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
    return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)


# --- negation ---

def _progress_negation(node, query, dag):
    excl_list = _get_fetched_exclusion_list(node)
    if excl_list is None:
        for action, obs in node.history:
            if action.tool == "get_exclusion_list" and not obs.ok:
                return 0.0
        return 0.1

    if not excl_list:
        return 0.0

    fact_slugs = {dag.get_node_by_name(f.concept) for f in node.state_facts}
    for slug in excl_list:
        if slug in fact_slugs:
            return 0.0    # a finding in the exclusion list is present → "No X" is false

    # Real gradient: how many exclusion items the agent has actually checked,
    # reaching 1.0 only when every item has been ruled out.
    checked = _exclusion_items_checked(node, excl_list)
    return 0.1 + (checked / len(excl_list)) * 0.9


def _exclusion_items_checked(node, excl_list):
    """How many exclusion items the agent has explicitly checked with an is_a."""
    checked = {
        action.args.get("node")
        for action, obs in node.history
        if action.tool == "is_a"
    }
    return sum(1 for slug in excl_list if slug in checked)


def _verify_negation(node, query, dag):
    path = list(node.history)
    excl_list = _get_fetched_exclusion_list(node)

    if excl_list is None:
        if node.answer:
            return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
        return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)

    fact_slugs = {dag.get_node_by_name(f.concept) for f in node.state_facts}
    for slug in excl_list:
        if slug in fact_slugs:
            return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)

    return SearchResult(
        answer=node.answer or f"No {query.target} found", tier="A", path=path,
        conf=1.0,
    )


# --- relational ---

def _progress_relational(node, query, dag):
    for action, obs in node.history:
        if action.tool in ("anatomy_of", "compose_laterality") and obs.ok:
            return 1.0
    return 0.2


def _verify_relational(node, query, dag):
    path = list(node.history)
    for action, obs in node.history:
        if action.tool in ("anatomy_of", "compose_laterality") and obs.ok:
            return SearchResult(
                answer=node.answer or str(obs.result), tier="A", path=path,
                conf=_min_conf(node),
            )
    if node.answer:
        return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
    return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)


# --- counting ---

def _verify_counting(node, query, dag):
    path = list(node.history)
    count = len({f.concept for f in node.state_facts})
    return SearchResult(
        answer=node.answer or str(count), tier="A", path=path, conf=1.0,
    )


# --- open ---

def _progress_open(node):
    return 0.5 if node.state_facts else 0.0


# --- shared_cause ---

def _progress_shared_cause(node, query, dag):
    a = query.constraints.get("finding_a", "")
    b = query.constraints.get("finding_b", "")
    shared = _find_shared_cause(node, a, b, dag)
    if shared:
        return 1.0
    explored = _explored_sides(node)
    if explored >= 2:
        return 0.6
    if explored >= 1:
        return 0.3
    return 0.1


def _verify_shared_cause(node, query, dag):
    path = list(node.history)
    a = query.constraints.get("finding_a", "")
    b = query.constraints.get("finding_b", "")
    shared = _find_shared_cause(node, a, b, dag)
    if shared:
        cause = shared[0]
        trace = [[cause, a], [cause, b]]
        answer = f"Yes, {cause}"
        return SearchResult(answer=answer, tier="A", path=path, conf=1.0)
    if _explored_sides(node) >= 2:
        return SearchResult(answer=node.answer or "No", tier="A", path=path, conf=0.8)
    if node.answer:
        return SearchResult(answer=node.answer, tier="B", path=path, conf=0.3)
    return SearchResult(answer="", tier="ABSTAIN", path=path, conf=0.0)


def _find_shared_cause(node, a, b, dag):
    """Find causes that appear in neighbors results for BOTH findings."""
    causes_a = set()
    causes_b = set()
    for action, obs in node.history:
        if action.tool == "neighbors" and obs.ok and obs.result:
            finding = action.args.get("node", "")
            causes = {c.lower() for c in obs.result}
            if finding.lower() == a.lower():
                causes_a |= causes
            elif finding.lower() == b.lower():
                causes_b |= causes
    overlap = causes_a & causes_b
    # Verify each candidate is a real shared cause on the KG
    verified = [c for c in overlap if dag.causal_edge(c, a) and dag.causal_edge(c, b)]
    return verified


def _explored_sides(node):
    """How many distinct findings have been explored with neighbors.
    Counts any neighbors call, even if it returned empty (still explored)."""
    findings = set()
    for action, obs in node.history:
        if action.tool == "neighbors":
            findings.add(action.args.get("node", "").lower())
    return len(findings)


# --- helpers ---

def _has_witness(node, dag):
    """True if a detected fact is-a the target via an is_a action.

    The is_a source must resolve to a detected fact — otherwise an ontology
    tautology like is_a(cardiomegaly, cardiac_abnormality), which is always true
    regardless of the image, would falsely witness a finding that was never seen.
    """
    fact_slugs = {dag.resolve_slug(f.concept) for f in node.state_facts}
    return any(
        action.tool == "is_a" and obs.ok and obs.result
        and dag.resolve_slug(action.args.get("node", "")) in fact_slugs
        for action, obs in node.history
    )


def _has_direct_match(node, query, dag):
    """True if any fact concept resolves to the query target (identity match)."""
    if not query.target:
        return False
    target_slug = dag.resolve_slug(query.target)
    return any(dag.resolve_slug(f.concept) == target_slug for f in node.state_facts)


def _resolve_target(query, dag):
    """Resolve query target to a DAG node id, or None if not in the graph."""
    if not query.target:
        return None
    node_id = dag.resolve_slug(query.target)
    return node_id if dag.get_node(node_id) else None


def _any_fact_is_a(node, target_id, dag):
    """True if any detected finding is, or is-a, the target."""
    for f in node.state_facts:
        fid = dag.resolve_slug(f.concept)
        if fid == target_id or dag.reachable_is_a(fid, target_id):
            return True
    return False


def _has_disjoint_violation(node):
    for action, obs in node.history:
        if action.tool == "disjoint" and obs.ok and obs.result is True:
            return True
    return False


def _get_fetched_exclusion_list(node):
    for action, obs in node.history:
        if action.tool == "get_exclusion_list" and obs.ok and obs.result is not None:
            return obs.result
    return None


def _min_conf(node):
    if not node.state_facts:
        return 1.0
    return min(f.conf for f in node.state_facts)
