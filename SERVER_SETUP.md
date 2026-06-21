# Server Setup — Full KRONOS Pipeline

Everything needed to run KRONOS end-to-end on a GPU server.

---

## 1. Clone & install dependencies

```bash
git clone <your-repo-url> KRONOS
cd KRONOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics transformers bitsandbytes accelerate
pip install open_clip_torch faiss-cpu
pip install pydantic numpy networkx pyyaml pillow pytest
```

---

## 2. Place model weights

You need two sets of weights:

```
weights/
  yolov12s_vindr.pt          # YOLO detector (trained on VinDr-CXR)
  llava-med-v1.5-mistral-7b/ # LLaVA-Med 1.5 (HuggingFace format)
```

BiomedCLIP downloads automatically on first use (~600MB from HuggingFace).

---

## 3. Place data

These should already be in the repo or transferred:

```
data/
  ontology/
    dag.yaml                  # ✅ already in repo
    exclusion_lists.yaml      # ✅ already in repo
    anatomy_zones.yaml        # ✅ already in repo
    synonyms.yaml             # ✅ already in repo
  vindr_cxr_vqa/
    vqa.json                  # ✅ already in repo
    train/                    # ~15,000 CXR images (PNG)
    test/                     # test split images
```

---

## 4. Build the RAG index

```bash
python scripts/build_rag_index.py \
    --vqa data/vindr_cxr_vqa/vqa.json \
    --images data/vindr_cxr_vqa/train \
    --out-index data/rag/vindr_index.faiss \
    --out-cases data/rag/vindr_cases.jsonl \
    --device cuda
```

Takes ~20-30 min on first run (downloads BiomedCLIP + encodes 15k images).
Produces:
- `data/rag/vindr_index.faiss` — FAISS vector index
- `data/rag/vindr_cases.jsonl` — case metadata (case_id, labels, reasons)

---

## 5. Set environment variables

```bash
export YOLO_WEIGHTS=weights/yolov12s_vindr.pt
export LLAVAMED_PATH=weights/llava-med-v1.5-mistral-7b
export TEST_IMAGE=data/vindr_cxr_vqa/train/000434271f63a053c4128a0ba6352c7f.png
```

---

## 6. Run tests

```bash
# CPU tests (should all pass, no GPU needed)
pytest tests/ -v --ignore=tests/test_integration_gpu.py

# GPU integration tests
pytest tests/test_integration_gpu.py -m gpu -v
```

Expected: all CPU tests pass, GPU tests pass if weights + TEST_IMAGE are set.

---

## 7. Run the full pipeline on a single image

```python
from src.ontology.dag import OntologyDAG
from src.perception.detector import Detector
from src.agent.llavamed import LLaVAMedAgent
from src.retrieval.encoder import load_encoder
from src.retrieval.index import RagIndex
from src.retrieval.retriever import Retriever
from src.pipeline import run
import json

# 1. Load ontology
dag = OntologyDAG(
    "data/ontology/dag.yaml",
    "data/ontology/exclusion_lists.yaml",
    "data/ontology/anatomy_zones.yaml",
)

# 2. Load detector
detector = Detector("weights/yolov12s_vindr.pt", dag=dag)

# 3. Load agent
agent = LLaVAMedAgent(model_path="weights/llava-med-v1.5-mistral-7b", quantize=True)

# 4. Load retriever
encoder = load_encoder(device="cuda")
cases = [json.loads(line) for line in open("data/rag/vindr_cases.jsonl")]
rag_index = RagIndex.load("data/rag/vindr_index.faiss", cases)
retriever = Retriever(rag_index, encoder=encoder)

# 5. Run on an image
from PIL import Image
image_path = "data/vindr_cxr_vqa/train/000434271f63a053c4128a0ba6352c7f.png"
image = Image.open(image_path).convert("RGB")
retriever.set_query_emb(encoder.encode(image))

result = run(
    image_path=image_path,
    question="Is there Cardiomegaly?",
    dag=dag,
    detector=detector,
    agent=agent,
    budget=20,
    retriever=retriever,
)

print(f"Answer: {result.answer}")
print(f"Tier:   {result.tier}")
print(f"Conf:   {result.conf}")
print(f"Path:   {len(result.path)} steps")
for action, obs in result.path:
    print(f"  {action.tool}({action.args}) -> ok={obs.ok}")
```

---

## 8. Run on multiple questions (batch eval)

```python
questions = [
    "Is there Cardiomegaly?",
    "Is there Pleural effusion?",
    "Where is the Nodule/Mass?",
    "How many findings are there?",
]

for q in questions:
    result = run(
        image_path=image_path,
        question=q,
        dag=dag,
        detector=detector,
        agent=agent,
        budget=20,
        retriever=retriever,
    )
    print(f"Q: {q}")
    print(f"A: {result.answer} (Tier {result.tier})\n")
```

---

## Quick checklist

- [ ] `pip install` all dependencies
- [ ] YOLO weights at `weights/yolov12s_vindr.pt`
- [ ] LLaVA-Med at `weights/llava-med-v1.5-mistral-7b/`
- [ ] VinDr train images at `data/vindr_cxr_vqa/train/`
- [ ] `python scripts/build_rag_index.py` completed
- [ ] `pytest tests/ -v` all green
- [ ] `pytest tests/test_integration_gpu.py -m gpu -v` all green
- [ ] Test single image pipeline run (step 7)
