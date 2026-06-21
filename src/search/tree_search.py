"""Best-first tree search guided by verifier-as-value.

search(query, facts, dag, agent, budget, k) → SearchResult

The agent proposes actions (or answers); tools execute them deterministically;
the verifier scores each node (closure-progress). Best-first expansion with
implicit backtracking via the frontier.
"""
from src.contracts import Action, Observation, TreeNode, SearchResult, Query, PerceptualFact
from src.tools.dispatch import run_tool
from src.tools.visual import fold_facts
from src.engine.verifier import closure_progress, verify
from src.ontology.dag import slugify

DEFAULT_IMG_WH = (2304, 2880)


def search(query, facts, dag, agent, budget=20, k=3, img_wh=None,
           image=None, detector_fn=None, vlm_fn=None, retriever=None):
    if img_wh is None:
        img_wh = DEFAULT_IMG_WH

    root = TreeNode(state_facts=list(facts), history=[])
    root = root.model_copy(update={"reward": closure_progress(root, query, dag)})

    frontier = [root]
    best_tier_b = None
    nodes_expanded = 0

    while frontier and nodes_expanded < budget:
        frontier.sort(key=lambda n: n.reward, reverse=True)
        node = frontier.pop(0)
        nodes_expanded += 1

        proposals = agent.propose_actions(node, query, k)

        for proposal in proposals:
            if isinstance(proposal, str):
                child = node.model_copy(update={
                    "answer": proposal,
                    "parent_id": id(node),
                })
                result = verify(child, query, dag)
                if result.tier == "A":
                    return result
                if result.tier == "B" and best_tier_b is None:
                    best_tier_b = result
                continue

            obs = run_tool(proposal, node.state_facts, dag, img_wh,
                          image=image, detector_fn=detector_fn, vlm_fn=vlm_fn,
                          retriever=retriever)
            new_history = list(node.history) + [(proposal, obs)]
            new_facts = list(node.state_facts)

            if proposal.kind == "visual" and obs.ok and obs.result:
                if proposal.tool == "re_detect" and isinstance(obs.result, list):
                    new_facts = fold_facts(new_facts, obs.result)
                elif proposal.tool == "inspect" and isinstance(obs.result, dict):
                    from src.contracts import PerceptualFact as PF
                    bbox = tuple(proposal.args.get("bbox", (0, 0, 0, 0)))
                    new_fact = PF(
                        concept=obs.result.get("concept", "unknown"),
                        bbox=bbox,
                        laterality="midline",
                        conf=obs.result.get("conf", 0.5),
                    )
                    new_facts = fold_facts(new_facts, [new_fact])

            child = TreeNode(
                state_facts=new_facts,
                history=new_history,
                parent_id=id(node),
            )
            child = child.model_copy(update={
                "reward": closure_progress(child, query, dag),
            })

            if child.reward >= 1.0:
                answer = _derive_answer(child, query)
                child = child.model_copy(update={"answer": answer})
                result = verify(child, query, dag)
                if result.tier == "A":
                    return result
                if result.tier == "B" and best_tier_b is None:
                    best_tier_b = result

            if child.reward > 0:
                frontier.append(child)

    if best_tier_b:
        return best_tier_b

    return SearchResult(answer="", tier="ABSTAIN", path=[], conf=0.0)


def _derive_answer(node, query):
    qtype = query.type

    if qtype == "existential":
        for action, obs in node.history:
            if action.tool == "is_a" and obs.ok and obs.result:
                return "Yes"
        # Direct match: fact concept matches target (identity, no is-a hop needed)
        target_slug = slugify(query.target)
        fact_slugs = {slugify(f.concept) for f in node.state_facts}
        if target_slug in fact_slugs:
            return "Yes"
        return "No"

    if qtype == "negation":
        return f"No {query.target} found"

    if qtype == "relational":
        for action, obs in node.history:
            if action.tool in ("anatomy_of", "compose_laterality") and obs.ok:
                return str(obs.result)
        return ""

    if qtype == "counting":
        count = len({f.concept for f in node.state_facts})
        return str(count)

    return ""
