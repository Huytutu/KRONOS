"""Unified tool dispatch — routes symbolic vs visual actions."""
from src.contracts import Action, Observation
from src.tools.symbolic import run_tool as run_symbolic
from src.tools.visual import run_inspect, run_re_detect, run_compare
from src.retrieval.tool import run_retrieve


def run_tool(action, facts, dag, img_wh, image=None, detector_fn=None, vlm_fn=None,
             retriever=None):
    if action.tool == "retrieve":
        return run_retrieve(action, retriever)
    if action.kind == "visual":
        return _run_visual(action, image, detector_fn, vlm_fn)
    return run_symbolic(action, facts, dag, img_wh)


def _run_visual(action, image, detector_fn, vlm_fn):
    tool = action.tool

    if tool == "inspect":
        if vlm_fn is None:
            return Observation(result=None, ok=False)
        return run_inspect(action, image, vlm_fn=vlm_fn)

    if tool == "re_detect":
        if detector_fn is None:
            return Observation(result=None, ok=False)
        return run_re_detect(action, image, detector_fn=detector_fn)

    if tool == "compare":
        if vlm_fn is None:
            return Observation(result=None, ok=False)
        return run_compare(action, image, vlm_fn=vlm_fn)

    return Observation(result=None, ok=False)
