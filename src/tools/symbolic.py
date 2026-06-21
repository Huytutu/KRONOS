"""Symbolic tool layer — thin wrappers turning OntologyDAG methods into Action → Observation."""
from src.contracts import Action, Observation


def run_tool(action, facts, dag, img_wh):
    tool = action.tool
    args = action.args

    if tool == "is_a":
        path = dag.reachable_is_a(args["node"], args["target"])
        if path:
            return Observation(result=path, ok=True)
        return Observation(result=None, ok=False)

    if tool == "disjoint":
        result = dag.disjoint(args["a"], args["b"])
        return Observation(result=result, ok=True)

    if tool == "anatomy_of":
        bbox = tuple(args["bbox"])
        zone = dag.anatomy_of(bbox, img_wh[0], img_wh[1])
        if zone:
            return Observation(result=zone, ok=True)
        return Observation(result=None, ok=False)

    if tool == "compose_laterality":
        bbox = tuple(args["bbox"])
        lat = dag.compose_laterality(bbox, img_wh[0], img_wh[1])
        return Observation(result=lat, ok=True)

    if tool == "get_exclusion_list":
        lst = dag.get_exclusion_list(args["name"])
        if lst is not None:
            return Observation(result=lst, ok=True)
        return Observation(result=None, ok=False)

    if tool == "retrieve":
        return Observation(result=None, ok=False)

    return Observation(result=None, ok=False)
