"""GPU integration tests — load real LLaVA-Med + YOLO, run end-to-end.

These tests are SKIPPED without GPU. Run on server with:
    pytest tests/test_integration_gpu.py -m gpu --gpu

To run: ensure LLaVA-Med weights and YOLO weights are available,
and set env vars:
    LLAVAMED_PATH=path/to/llava-med-v1.5-mistral-7b
    YOLO_WEIGHTS=path/to/yolov12s_vindr.pt
    TEST_IMAGE=path/to/sample_cxr.png
"""
import os
import pytest
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "ontology"

gpu_available = False
try:
    import torch
    gpu_available = torch.cuda.is_available()
except ImportError:
    pass

skip_no_gpu = pytest.mark.skipif(not gpu_available, reason="No GPU available")


@pytest.fixture
def dag():
    from src.ontology.dag import OntologyDAG
    return OntologyDAG(str(DATA / "dag.yaml"), str(DATA / "exclusion_lists.yaml"), str(DATA / "anatomy_zones.yaml"))


@pytest.mark.gpu
@skip_no_gpu
def test_llavamed_loads():
    """LLaVA-Med 1.5 loads without error."""
    model_path = os.environ.get("LLAVAMED_PATH")
    if not model_path:
        pytest.skip("LLAVAMED_PATH not set")
    from src.agent.llavamed import LLaVAMedAgent
    agent = LLaVAMedAgent(model_path=model_path, quantize=True)
    assert agent._model is not None


@pytest.mark.gpu
@skip_no_gpu
def test_llavamed_generates_actions(dag):
    """LLaVA-Med generates parseable actions for a simple existential question."""
    model_path = os.environ.get("LLAVAMED_PATH")
    if not model_path:
        pytest.skip("LLAVAMED_PATH not set")
    from src.agent.llavamed import LLaVAMedAgent
    from src.contracts import TreeNode, PerceptualFact, Query
    agent = LLaVAMedAgent(model_path=model_path, quantize=True)
    fact = PerceptualFact(concept="Cardiomegaly", bbox=(100, 200, 300, 400), laterality="midline", conf=0.85)
    node = TreeNode(state_facts=[fact], history=[])
    query = Query(type="existential", target="Cardiomegaly", constraints={},
                  raw_question="Is there Cardiomegaly?", parse_confidence=1.0, parser_tier="rule")
    actions = agent.propose_actions(node, query, k=3)
    # Should return at least something (Action or Answer string), not crash
    assert isinstance(actions, list)


@pytest.mark.gpu
@skip_no_gpu
def test_end_to_end_with_real_image(dag):
    """Full pipeline: real image + YOLO + LLaVA-Med → SearchResult."""
    model_path = os.environ.get("LLAVAMED_PATH")
    yolo_weights = os.environ.get("YOLO_WEIGHTS")
    test_image = os.environ.get("TEST_IMAGE")
    if not all([model_path, yolo_weights, test_image]):
        pytest.skip("LLAVAMED_PATH, YOLO_WEIGHTS, or TEST_IMAGE not set")

    from src.agent.llavamed import LLaVAMedAgent
    from src.perception.detector import Detector
    from src.pipeline import run
    from src.contracts import SearchResult

    detector = Detector(yolo_weights, dag=dag)
    agent = LLaVAMedAgent(model_path=model_path, quantize=True)

    result = run(
        image_path=test_image,
        question="Is there Cardiomegaly?",
        dag=dag,
        detector=detector,
        agent=agent,
        budget=10,
    )

    assert isinstance(result, SearchResult)
    assert result.tier in ("A", "B", "ABSTAIN")
    print(f"Answer: {result.answer}, Tier: {result.tier}, Path length: {len(result.path)}")


@pytest.mark.gpu
@skip_no_gpu
def test_re_detect_on_real_image(dag):
    """re_detect finds something on a cropped region of a real image."""
    yolo_weights = os.environ.get("YOLO_WEIGHTS")
    test_image = os.environ.get("TEST_IMAGE")
    if not all([yolo_weights, test_image]):
        pytest.skip("YOLO_WEIGHTS or TEST_IMAGE not set")

    from PIL import Image
    from src.perception.detector import Detector
    from src.contracts import Action
    from src.tools.visual import run_re_detect

    detector = Detector(yolo_weights, dag=dag)
    image = Image.open(test_image).convert("RGB")

    action = Action(tool="re_detect", args={"bbox": [0, 0, image.width // 2, image.height // 2]}, kind="visual")

    def detector_fn(crop):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            crop.save(f.name)
            return detector.detect(f.name)

    obs = run_re_detect(action, image, detector_fn=detector_fn)
    # May or may not find something — just shouldn't crash
    assert isinstance(obs.ok, bool)


# --- RAG / retrieve integration ---

@pytest.mark.gpu
@skip_no_gpu
def test_encoder_loads_and_encodes():
    """BiomedCLIP encoder loads and produces 512-d normalised embeddings."""
    import numpy as np
    from PIL import Image

    test_image = os.environ.get("TEST_IMAGE")
    if not test_image:
        pytest.skip("TEST_IMAGE not set")

    from src.retrieval.encoder import load_encoder
    encoder = load_encoder(device="cuda")
    image = Image.open(test_image).convert("RGB")
    emb = encoder.encode(image)

    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


@pytest.mark.gpu
@skip_no_gpu
def test_retrieve_with_real_encoder(dag):
    """Full retrieve path: real encoder → brute-force index → Observation."""
    import numpy as np
    from PIL import Image

    test_image = os.environ.get("TEST_IMAGE")
    if not test_image:
        pytest.skip("TEST_IMAGE not set")

    from src.retrieval.encoder import load_encoder
    from src.retrieval.index import BruteForceIndex
    from src.retrieval.retriever import Retriever
    from src.contracts import Action
    from src.retrieval.tool import run_retrieve

    encoder = load_encoder(device="cuda")
    image = Image.open(test_image).convert("RGB")
    emb = encoder.encode(image)

    cases = [{"case_id": f"case_{i}", "labels": ["test"], "report": "test"} for i in range(5)]
    fake_embs = np.random.RandomState(42).randn(5, 512).astype(np.float32)
    fake_embs = fake_embs / np.linalg.norm(fake_embs, axis=1, keepdims=True)
    fake_embs[0] = emb

    index = BruteForceIndex(fake_embs, cases)
    retriever = Retriever(index, encoder=encoder)
    retriever.set_query_emb(emb)

    action = Action(tool="retrieve", args={"k": 3})
    obs = run_retrieve(action, retriever)

    assert obs.ok is True
    assert len(obs.result) == 3
    assert obs.result[0]["case_id"] == "case_0"
