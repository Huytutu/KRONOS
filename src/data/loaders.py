"""Dataset loaders — one per dataset, all yielding the same QAItem shape.

Each dataset stores its questions differently (a JSONL of shared-cause pairs, a
nested VQA JSON, a multiple-choice benchmark). A loader hides that difference:
it reads the raw file and returns a flat list of QAItem, so the rest of the code
works the same way no matter which dataset it came from.

The core fields (id, image, question, answer) are what every dataset has in
common; anything dataset-specific lives in `meta`.
"""
from dataclasses import dataclass, field
import json


@dataclass
class QAItem:
    id: str
    dataset: str            # "multihop" | "vindr_vqa" | "chestagentbench"
    image: str              # primary image path, relative to the repo root
    question: str
    answer: str
    meta: dict = field(default_factory=dict)   # dataset-specific extras


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pick(record, keys):
    """Copy the keys that exist into a new dict (dataset-specific meta)."""
    return {k: record[k] for k in keys if k in record}


def load_multihop(path):
    """Shared-cause Yes/No QA (data/multihop_qa/qa.jsonl)."""
    meta_keys = ("finding_a", "finding_b", "gold_causes",
                 "support_edges", "hops", "single_cause")
    items = []
    for r in _read_jsonl(path):
        items.append(QAItem(
            id=r["id"], dataset="multihop", image=r["image"],
            question=r["question"], answer=r["answer"],
            meta=_pick(r, meta_keys),
        ))
    return items


def load_vindr_vqa(path, image_dir="data/vindr_cxr_vqa/train"):
    """VinDr-CXR-VQA (data/vindr_cxr_vqa/vqa.json): one QAItem per question.
    A single study holds several questions, so we flatten them."""
    meta_keys = ("type", "difficulty", "gt_finding", "gt_location", "reason")
    with open(path, encoding="utf-8") as f:
        studies = json.load(f)
    items = []
    for study in studies:
        image_id = study["image_id"]
        image = f"{image_dir}/{image_id}.png"
        for i, qa in enumerate(study["vqa"]):
            items.append(QAItem(
                id=f"{image_id}_{i}", dataset="vindr_vqa", image=image,
                question=qa["question"], answer=qa["answer"],
                meta=_pick(qa, meta_keys),
            ))
    return items


def load_slake(path, image_dir="data/Slake1.0/imgs", modality="X-Ray", lang="en"):
    """SLAKE 1.0: one QAItem per question, filtered by modality and language."""
    meta_keys = ("content_type", "answer_type", "triple", "location", "modality", "img_id")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    items = []
    for r in records:
        if r.get("modality") != modality:
            continue
        if r.get("q_lang") != lang:
            continue
        image = f"{image_dir}/{r['img_name']}"
        items.append(QAItem(
            id=f"{r['img_name'].split('/')[0]}_{r['qid']}",
            dataset="slake", image=image,
            question=r["question"], answer=str(r["answer"]),
            meta=_pick(r, meta_keys),
        ))
    return items


def load_chestagentbench(path, figures_root="data/chestagentbench"):
    """ChestAgentBench (data/chestagentbench/metadata.jsonl): multiple-choice
    clinical questions over one or more figures."""
    meta_keys = ("explanation", "type", "categories", "sections", "case_id")
    items = []
    for r in _read_jsonl(path):
        images = [f"{figures_root}/{p}" for p in r.get("images", [])]
        meta = _pick(r, meta_keys)
        meta["images"] = images
        items.append(QAItem(
            id=r["full_question_id"], dataset="chestagentbench",
            image=images[0] if images else "",
            question=r["question"], answer=r["answer"],
            meta=meta,
        ))
    return items


LOADERS = {
    "multihop": load_multihop,
    "vindr_vqa": load_vindr_vqa,
    "slake": load_slake,
    "chestagentbench": load_chestagentbench,
}
