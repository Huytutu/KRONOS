"""Tool wrapper for retrieve — returns Observation."""
from src.contracts import Observation


def run_retrieve(action, retriever):
    if retriever is None:
        return Observation(result=None, ok=False)
    k = action.args.get("k", 5)
    cases = retriever.retrieve(k)
    if not cases:
        return Observation(result=None, ok=False)
    return Observation(result=cases, ok=True)
