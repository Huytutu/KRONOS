"""End-to-end pipeline — ties perception, parsing, agent, and tree search."""
from src.contracts import SearchResult, Query
from src.search.tree_search import search
from src.agent.mock import MockAgent
from src.question.parser import QuestionParser

VINDR_FINDINGS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity",
    "Nodule/Mass", "Other lesion", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
]

_parser = QuestionParser(finding_vocab=VINDR_FINDINGS)


def run_with_facts(question, facts, dag, agent=None, budget=20, k=3, img_wh=None):
    """Run pipeline with pre-computed facts (no detector needed)."""
    if agent is None:
        agent = MockAgent()

    query = _parser.parse(question)
    return search(query, facts, dag, agent, budget=budget, k=k, img_wh=img_wh)


def run(image_path, question, dag, detector, agent, budget=20, k=3):
    """Full end-to-end: load image → detect → parse → search → result.

    Requires detector (YOLO) and agent (LLaVA-Med or Mock).
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    img_wh = image.size

    facts = detector.detect(str(image_path))

    query = _parser.parse(question)

    if hasattr(agent, "set_image"):
        agent.set_image(image)

    vlm_fn = None
    detector_fn = None
    if hasattr(agent, "_inference_fn") and agent._inference_fn:
        vlm_fn = lambda crop, prompt: agent._inference_fn(prompt, crop)
    if hasattr(detector, "model"):
        detector_fn = lambda crop: detector._results_to_facts(detector.model(crop, verbose=False))

    return search(
        query, facts, dag, agent, budget=budget, k=k,
        img_wh=img_wh, image=image, detector_fn=detector_fn, vlm_fn=vlm_fn,
    )
