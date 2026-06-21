from src.contracts import Candidate


class EPerc:
    """Proposer that answers directly from perception facts via name-match.

    No DAG, no reasoning. Fast but cannot handle abstract categories
    (e.g. "cardiac abnormality" when facts have "Cardiomegaly").
    """

    def propose(self, query, facts):
        qtype = query.type

        if qtype == "existential":
            return self._existential(query, facts)
        elif qtype == "negation":
            return self._negation(query, facts)
        elif qtype == "relational":
            return self._relational(query, facts)
        elif qtype == "counting":
            return self._counting(facts)
        return None

    def _existential(self, query, facts):
        # Open query: list all findings
        if query.constraints.get("scope") == "open":
            names = [f.concept for f in facts]
            answer = ", ".join(names) if names else "No findings"
            conf = min((f.conf for f in facts), default=1.0)
            anchor = [(f.conf, f.bbox) for f in facts]
            return Candidate(answer=answer, anchor=anchor, head_id="E_perc", conf=conf)

        matches = [f for f in facts if f.concept == query.target]
        if matches:
            best = max(matches, key=lambda f: f.conf)
            anchor = [(f.conf, f.bbox) for f in matches]
            return Candidate(answer="Yes", anchor=anchor, head_id="E_perc", conf=best.conf)

        return Candidate(answer="No", anchor=[], head_id="E_perc", conf=1.0)

    def _negation(self, query, facts):
        matches = [f for f in facts if f.concept == query.target]
        if matches:
            best = max(matches, key=lambda f: f.conf)
            anchor = [(f.conf, f.bbox) for f in matches]
            return Candidate(
                answer=f"{query.target} is present",
                anchor=anchor, head_id="E_perc", conf=best.conf,
            )

        return Candidate(
            answer=f"No {query.target} found",
            anchor=[], head_id="E_perc", conf=1.0,
        )

    def _relational(self, query, facts):
        matches = [f for f in facts if f.concept == query.target]
        if not matches:
            return None

        best = max(matches, key=lambda f: f.conf)
        attr = query.constraints.get("attr")

        if attr == "laterality":
            answer = best.laterality
        elif attr == "location":
            x1, y1, x2, y2 = best.bbox
            answer = f"bbox ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        else:
            answer = best.laterality

        return Candidate(
            answer=answer, anchor=[(best.conf, best.bbox)],
            head_id="E_perc", conf=best.conf,
        )

    def _counting(self, facts):
        count = len({f.concept for f in facts})
        anchor = [(f.conf, f.bbox) for f in facts]
        return Candidate(answer=str(count), anchor=anchor, head_id="E_perc", conf=1.0)
