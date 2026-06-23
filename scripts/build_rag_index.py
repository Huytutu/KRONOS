"""Build FAISS index + cases.jsonl from VinDr-CXR train split.

Usage:
    python scripts/build_rag_index.py \
        --vqa data/vindr_cxr_vqa/vqa.json \
        --images data/vindr_cxr_vqa/train \
        --out-index data/rag/vindr_index.faiss \
        --out-cases data/rag/vindr_cases.jsonl
"""
import argparse
import json
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_cases(vqa_path, image_dir):
    """Group VQA entries by image_id, keep only train-split images."""
    with open(vqa_path) as f:
        records = json.load(f)

    image_dir = Path(image_dir)
    cases = []

    for record in records:
        image_id = record["image_id"]
        image_path = image_dir / f"{image_id}.png"
        if not image_path.exists():
            continue

        labels = sorted(set(qa["gt_finding"] for qa in record["vqa"]))
        reasons = sorted(set(qa["reason"] for qa in record["vqa"] if qa.get("reason")))
        report = " ".join(reasons)

        cases.append({
            "case_id": image_id,
            "labels": labels,
            "report": report,
            "image_path": str(image_path),
        })

    return cases


def encode_cases(cases, encoder):
    from PIL import Image

    embeddings = []
    for i, case in enumerate(cases):
        image = Image.open(case["image_path"]).convert("RGB")
        emb = encoder.encode(image)
        embeddings.append(emb)
        if (i + 1) % 100 == 0:
            print(f"  encoded {i + 1}/{len(cases)}")

    return np.array(embeddings, dtype=np.float32)


def build_index(embeddings):
    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from VinDr-CXR")
    parser.add_argument("--vqa", default="data/vindr_cxr_vqa/vqa.json")
    parser.add_argument("--images", default="data/vindr_cxr_vqa/train")
    parser.add_argument("--out-index", default="data/rag/vindr_index.faiss")
    parser.add_argument("--out-cases", default="data/rag/vindr_cases.jsonl")
    parser.add_argument("--encoder", default="weights/BiomedCLIP")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Building cases from VQA...")
    cases = build_cases(args.vqa, args.images)
    print(f"  {len(cases)} train cases")

    print("Loading encoder...")
    from src.retrieval.encoder import load_encoder
    encoder = load_encoder(model_path=args.encoder, device=args.device)

    print("Encoding images...")
    embeddings = encode_cases(cases, encoder)

    print("Building FAISS index...")
    index = build_index(embeddings)

    out_index = Path(args.out_index)
    out_cases = Path(args.out_cases)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_cases.parent.mkdir(parents=True, exist_ok=True)

    import faiss
    faiss.write_index(index, str(out_index))

    stripped = [{"case_id": c["case_id"], "labels": c["labels"], "report": c["report"]}
                for c in cases]
    with open(out_cases, "w") as f:
        for case in stripped:
            f.write(json.dumps(case) + "\n")

    print(f"Saved {out_index} ({index.ntotal} vectors) + {out_cases}")


if __name__ == "__main__":
    main()
