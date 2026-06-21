"""Tests for unified tool dispatch — routes symbolic vs visual."""
import pytest
from pathlib import Path
from PIL import Image
from src.contracts import Action, PerceptualFact
from src.ontology.dag import OntologyDAG

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"


@pytest.fixture
def dag():
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (512, 512), color="gray")


def test_dispatch_symbolic(dag):
    from src.tools.dispatch import run_tool
    action = Action(tool="is_a", args={"node": "cardiomegaly", "target": "cardiac_abnormality"})
    obs = run_tool(action, [], dag, (2304, 2880))
    assert obs.ok is True
    assert "cardiomegaly" in obs.result


def test_dispatch_visual_inspect(dag, dummy_image):
    from src.tools.dispatch import run_tool
    def mock_vlm(crop, prompt):
        return '{"concept": "consolidation", "conf": 0.7, "description": "opacity"}'
    action = Action(tool="inspect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_tool(action, [], dag, (512, 512), image=dummy_image, vlm_fn=mock_vlm)
    assert obs.ok is True


def test_dispatch_visual_no_image(dag):
    from src.tools.dispatch import run_tool
    action = Action(tool="inspect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    obs = run_tool(action, [], dag, (512, 512), image=None)
    assert obs.ok is False


def test_dispatch_re_detect(dag, dummy_image):
    from src.tools.dispatch import run_tool
    def mock_detector(crop):
        return [PerceptualFact(concept="Nodule/Mass", bbox=(10, 10, 50, 50), laterality="right", conf=0.72)]
    action = Action(tool="re_detect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_tool(action, [], dag, (512, 512), image=dummy_image, detector_fn=mock_detector)
    assert obs.ok is True


def test_dispatch_compare(dag, dummy_image):
    from src.tools.dispatch import run_tool
    def mock_vlm(c1, c2, prompt):
        return '{"comparison": "left darker", "laterality_hint": "left"}'
    action = Action(tool="compare", args={"bbox1": [0, 0, 256, 256], "bbox2": [256, 0, 512, 256]}, kind="visual")
    obs = run_tool(action, [], dag, (512, 512), image=dummy_image, vlm_fn=mock_vlm)
    assert obs.ok is True


def test_dispatch_unknown_visual(dag, dummy_image):
    from src.tools.dispatch import run_tool
    action = Action(tool="inspect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    obs = run_tool(action, [], dag, (512, 512), image=dummy_image)
    assert obs.ok is False
