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
    return 0.0


def verify(node, query, dag):
    qtype = query.type
    path = list(node.history)

    if qtype == "open":
        return SearchResult(
            answer=node.answer or "", tier="B", path=path,
            conf=_min_conf(node),
        )

    if qtype == "existential":
        return _verify_existential(node, query, dag)
    if qtype == "negation":
        return _verify_negation(node, query, dag)
    if qtype == "relational":
        return _verify_relational(node, query, dag)
    if qtype == "counting":
        return _verify_counting(node, query, dag)

    return SearchResult(answer=node.answer or "", tier="ABSTAIN", path=path, conf=0.0)


# --- existential ---

def _progress_existential(node, query, dag):
    if _has_witness(node) or _has_direct_match(node, query, dag):
        return 1.0
    target_slug = dag.get_node_by_name(query.target) if query.target else None
    if target_slug and dag.get_node(target_slug):
        return 0.2
    return 0.1


def _verify_existential(node, query, dag):
    path = list(node.history)
    if _has_witness(node) or _has_direct_match(node, query, dag):
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

    fact_slugs = {dag.get_node_by_name(f.concept) for f in node.state_facts}
    for slug in excl_list:
        if slug in fact_slugs:
            return 0.0

    checked = len(excl_list)
    total = len(excl_list)
    if total == 0:
        return 0.0
    return checked / total


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


# --- helpers ---

def _has_witness(node):
    """True if any is_a action in history found a path (existential witness)."""
    return any(
        action.tool == "is_a" and obs.ok and obs.result
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
