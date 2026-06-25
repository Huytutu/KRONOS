"""MockAgent — scripted actions per question type, for deterministic tests."""
from src.contracts import Action, TreeNode, Query
from src.ontology.dag import slugify


class MockAgent:
    """Returns scripted actions based on question type and current state.

    Enough to drive tree search tests without any neural model.
    """

    def propose_actions(self, node, query, k=3):
        qtype = query.type

        if qtype == "existential":
            return self._existential(node, query, k)
        if qtype == "negation":
            return self._negation(node, query, k)
        if qtype == "relational":
            return self._relational(node, query, k)
        if qtype == "counting":
            return self._counting(node, query, k)
        if qtype == "open":
            return self._open(node, query, k)
        if qtype == "shared_cause":
            return self._shared_cause(node, query, k)
        return []

    def _existential(self, node, query, k):
        target_slug = slugify(query.target) or "unknown"
        checked = {
            a.args.get("node")
            for a, _ in node.history
            if a.tool == "is_a"
        }

        results = []
        for fact in node.state_facts:
            slug = slugify(fact.concept)
            if slug not in checked:
                results.append(Action(
                    tool="is_a",
                    args={"node": slug, "target": target_slug},
                ))
                if len(results) >= k:
                    break

        if results:
            return results

        # Every finding has been checked against the target and none is-a it.
        # Under the closed-world assumption (perception is exhaustive), the
        # target is absent → answer "No". The verifier re-confirms on the DAG.
        return ["No"]

    def _negation(self, node, query, k):
        has_exclusion = any(
            a.tool == "get_exclusion_list" for a, _ in node.history
        )

        if not has_exclusion:
            return [Action(
                tool="get_exclusion_list",
                args={"name": query.target},
            )]

        excl_list = None
        for action, obs in node.history:
            if action.tool == "get_exclusion_list" and obs.ok:
                excl_list = obs.result
                break

        if excl_list is None:
            return [f"No {query.target} found"]

        checked = {
            a.args.get("node")
            for a, _ in node.history
            if a.tool == "is_a"
        }
        results = []
        for slug in excl_list:
            if slug not in checked:
                results.append(Action(
                    tool="is_a",
                    args={"node": slug, "target": slug},
                ))
                if len(results) >= k:
                    break

        if not results:
            return [f"No {query.target} found"]
        return results

    def _relational(self, node, query, k):
        attr = query.constraints.get("attr", "location")
        tool = "compose_laterality" if attr == "laterality" else "anatomy_of"

        fact = self._fact_to_localize(node, query)
        if fact is None:
            return []
        return [Action(tool=tool, args={"bbox": list(fact.bbox)})]

    def _fact_to_localize(self, node, query):
        """Finding whose location we report. Match the named target, or fall
        back to the first finding when the question names none (e.g.
        'Which side shows the abnormality?', where target is None)."""
        if query.target:
            for fact in node.state_facts:
                if slugify(fact.concept) == slugify(query.target):
                    return fact
            return None
        return node.state_facts[0] if node.state_facts else None

    def _counting(self, node, query, k):
        count = len({f.concept for f in node.state_facts})
        return [str(count)]

    def _shared_cause(self, node, query, k):
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

    def _open(self, node, query, k):
        names = [f.concept for f in node.state_facts]
        return [", ".join(names) if names else "No findings"]
