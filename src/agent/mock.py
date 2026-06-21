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
        return []

    def _existential(self, node, query, k):
        results = []
        target_slug = slugify(query.target) or "unknown"

        for fact in node.state_facts:
            slug = slugify(fact.concept)
            results.append(Action(
                tool="is_a",
                args={"node": slug, "target": target_slug},
            ))
            if len(results) >= k:
                break

        if not results:
            results.append(Action(
                tool="is_a",
                args={"node": "unknown", "target": target_slug},
            ))
        return results

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
        results = []

        for fact in node.state_facts:
            if slugify(fact.concept) == slugify(query.target):
                bbox = list(fact.bbox)
                if attr == "laterality":
                    results.append(Action(
                        tool="compose_laterality", args={"bbox": bbox},
                    ))
                else:
                    results.append(Action(
                        tool="anatomy_of", args={"bbox": bbox},
                    ))
                break

        if not results:
            results.append(Action(
                tool="anatomy_of",
                args={"bbox": [0, 0, 100, 100]},
            ))
        return results

    def _counting(self, node, query, k):
        count = len({f.concept for f in node.state_facts})
        return [str(count)]

    def _open(self, node, query, k):
        names = [f.concept for f in node.state_facts]
        return [", ".join(names) if names else "No findings"]
