"""Tests for visual tools — inspect, re_detect, compare. All VLM/detector mocked."""
import pytest
from PIL import Image
from src.contracts import Action, Observation, PerceptualFact


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (512, 512), color="gray")


def _fact(concept, bbox=(100, 200, 300, 400), lat="midline", conf=0.85):
    return PerceptualFact(concept=concept, bbox=bbox, laterality=lat, conf=conf)


# --- inspect ---

def test_inspect_parses_finding(dummy_image):
    from src.tools.visual import run_inspect
    def mock_vlm_describe(crop, prompt):
        return '{"concept": "consolidation", "conf": 0.75, "description": "dense opacity"}'
    action = Action(tool="inspect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_inspect(action, dummy_image, vlm_fn=mock_vlm_describe)
    assert obs.ok is True
    assert obs.result["concept"] == "consolidation"


def test_inspect_malformed_vlm(dummy_image):
    from src.tools.visual import run_inspect
    def mock_vlm_bad(crop, prompt):
        return "I see something weird here"
    action = Action(tool="inspect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_inspect(action, dummy_image, vlm_fn=mock_vlm_bad)
    assert obs.ok is False


def test_inspect_no_image():
    from src.tools.visual import run_inspect
    action = Action(tool="inspect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    obs = run_inspect(action, None, vlm_fn=lambda c, p: "")
    assert obs.ok is False


# --- re_detect ---

def test_re_detect_returns_facts(dummy_image):
    from src.tools.visual import run_re_detect
    def mock_detector(crop):
        return [_fact("Nodule/Mass", bbox=(10, 10, 50, 50), conf=0.72)]
    action = Action(tool="re_detect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_re_detect(action, dummy_image, detector_fn=mock_detector)
    assert obs.ok is True
    assert len(obs.result) >= 1
    assert obs.result[0].concept == "Nodule/Mass"


def test_re_detect_maps_bbox_back(dummy_image):
    from src.tools.visual import run_re_detect
    def mock_detector(crop):
        return [_fact("Nodule/Mass", bbox=(10, 20, 50, 60), conf=0.72)]
    action = Action(tool="re_detect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_re_detect(action, dummy_image, detector_fn=mock_detector)
    assert obs.ok is True
    fact = obs.result[0]
    assert fact.bbox[0] >= 100
    assert fact.bbox[1] >= 100


def test_re_detect_nothing_found(dummy_image):
    from src.tools.visual import run_re_detect
    def mock_detector(crop):
        return []
    action = Action(tool="re_detect", args={"bbox": [100, 100, 300, 300]}, kind="visual")
    obs = run_re_detect(action, dummy_image, detector_fn=mock_detector)
    assert obs.ok is False


def test_re_detect_no_image():
    from src.tools.visual import run_re_detect
    action = Action(tool="re_detect", args={"bbox": [0, 0, 100, 100]}, kind="visual")
    obs = run_re_detect(action, None, detector_fn=lambda c: [])
    assert obs.ok is False


# --- compare ---

def test_compare_parses_result(dummy_image):
    from src.tools.visual import run_compare
    def mock_vlm_compare(crop1, crop2, prompt):
        return '{"comparison": "left shows more opacity", "laterality_hint": "left"}'
    action = Action(tool="compare", args={"bbox1": [0, 0, 256, 256], "bbox2": [256, 0, 512, 256]}, kind="visual")
    obs = run_compare(action, dummy_image, vlm_fn=mock_vlm_compare)
    assert obs.ok is True
    assert "left" in obs.result["comparison"]


def test_compare_malformed(dummy_image):
    from src.tools.visual import run_compare
    def mock_vlm_bad(crop1, crop2, prompt):
        return "both look the same to me"
    action = Action(tool="compare", args={"bbox1": [0, 0, 256, 256], "bbox2": [256, 0, 512, 256]}, kind="visual")
    obs = run_compare(action, dummy_image, vlm_fn=mock_vlm_bad)
    assert obs.ok is False


# --- fact folding ---

def test_fold_deduplicates_by_iou():
    from src.tools.visual import fold_facts
    existing = [_fact("Consolidation", bbox=(100, 100, 300, 300))]
    new_facts = [_fact("Consolidation", bbox=(110, 110, 290, 290), conf=0.9)]
    merged = fold_facts(existing, new_facts, iou_threshold=0.5)
    assert len(merged) == 1


def test_fold_adds_non_overlapping():
    from src.tools.visual import fold_facts
    existing = [_fact("Consolidation", bbox=(100, 100, 300, 300))]
    new_facts = [_fact("Nodule/Mass", bbox=(400, 400, 450, 450), conf=0.72)]
    merged = fold_facts(existing, new_facts, iou_threshold=0.5)
    assert len(merged) == 2
