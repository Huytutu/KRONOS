"""End-to-end pipeline — ties perception, parsing, agent, and tree search."""
import os
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


def _make_parser():
    llm_client = None
    if os.environ.get("GROQ_API_KEY"):
        from src.llm.groq_client import complete
        llm_client = complete
    return QuestionParser(finding_vocab=VINDR_FINDINGS, llm_client=llm_client)


_parser = _make_parser()


def run_with_facts(question, facts, dag, agent=None, budget=20, k=3, img_wh=None,
                   retriever=None):
    """Run pipeline with pre-computed facts (no detector needed)."""
    if agent is None:
        agent = MockAgent()

    query = _parser.parse(question)
    return search(query, facts, dag, agent, budget=budget, k=k, img_wh=img_wh,
                  retriever=retriever)


def run(image_path, question, dag, detector, agent, budget=20, k=3, retriever=None):
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

    # Give the retriever this image's embedding so the retrieve tool can find
    # similar cases; without it retriever.query_emb stays None and retrieve
    # returns nothing.
    if retriever is not None and getattr(retriever, "encoder", None) is not None:
        retriever.set_query_emb(retriever.encoder.encode(image))

    vlm_fn = None
    detector_fn = None
    if hasattr(agent, "_inference_fn") and agent._inference_fn:
        vlm_fn = lambda crop, prompt: agent._inference_fn(prompt, crop)
    if hasattr(detector, "model"):
        detector_fn = lambda crop: detector._results_to_facts(detector.model(crop, verbose=False))

    return search(
        query, facts, dag, agent, budget=budget, k=k,
        img_wh=img_wh, image=image, detector_fn=detector_fn, vlm_fn=vlm_fn,
        retriever=retriever,
    )
