"""Tests for Detector (YOLOv12s) and PerceptionOracle (GT from train.csv)."""

import pytest
from src.contracts import PerceptualFact

WEIGHTS_PATH = "weights/yolov12s_vindr.pt"
TRAIN_CSV = "data/vindr_cxr_vqa/train.csv"
TRAIN_DIR = "data/vindr_cxr_vqa/train"
DAG_PATH = "data/ontology/dag.yaml"
ZONES_PATH = "data/ontology/anatomy_zones.yaml"

VINDR_NAMES = {
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
}

VALID_LATERALITIES = {"left", "right", "bilateral", "midline"}

# Known image with Cardiomegaly (from vqa.json)
KNOWN_IMAGE_ID = "9a5094b2563a1ef3ff50dc5c7ff71345"


# ============================================================
# Task 2: Detector tests
# ============================================================

class TestDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        from src.perception.detector import Detector
        return Detector(WEIGHTS_PATH)

    def test_loads(self, detector):
        assert detector is not None

    def test_inference_returns_list(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        assert isinstance(facts, list)

    def test_facts_are_perceptual_facts(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        for f in facts:
            assert isinstance(f, PerceptualFact)

    def test_concept_in_vindr_names(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        for f in facts:
            assert f.concept in VINDR_NAMES, f"Unknown concept: {f.concept}"

    def test_conf_in_range(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        for f in facts:
            assert 0.0 <= f.conf <= 1.0

    def test_valid_laterality(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        for f in facts:
            assert f.laterality in VALID_LATERALITIES

    def test_bbox_is_4_tuple(self, detector):
        img = f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"
        facts = detector.detect(img)
        for f in facts:
            assert len(f.bbox) == 4

    def test_detect_batch(self, detector):
        imgs = [f"{TRAIN_DIR}/{KNOWN_IMAGE_ID}.png"]
        results = detector.detect_batch(imgs)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], list)


# ============================================================
# Task 4: Oracle tests
# ============================================================

class TestOracle:
    @pytest.fixture(scope="class")
    def oracle(self):
        from src.perception.oracle import PerceptionOracle
        from src.ontology.dag import OntologyDAG
        dag = OntologyDAG(DAG_PATH, zones_path=ZONES_PATH)
        return PerceptionOracle(TRAIN_CSV, dag)

    def test_loads(self, oracle):
        assert oracle is not None

    def test_known_image_returns_facts(self, oracle):
        facts = oracle.get_facts(KNOWN_IMAGE_ID)
        assert len(facts) > 0

    def test_facts_are_perceptual_facts(self, oracle):
        facts = oracle.get_facts(KNOWN_IMAGE_ID)
        for f in facts:
            assert isinstance(f, PerceptualFact)

    def test_conf_is_one(self, oracle):
        facts = oracle.get_facts(KNOWN_IMAGE_ID)
        for f in facts:
            assert f.conf == 1.0

    def test_no_finding_image_returns_empty(self, oracle):
        # Image with only "No finding" annotations
        ids = oracle.get_all_image_ids()
        # Find an image NOT in the finding set
        import csv
        all_ids = set()
        with open(TRAIN_CSV) as f:
            for r in csv.DictReader(f):
                all_ids.add(r["image_id"])
        no_finding_ids = all_ids - ids
        if no_finding_ids:
            nf_id = next(iter(no_finding_ids))
            facts = oracle.get_facts(nf_id)
            assert len(facts) == 0

    def test_concept_in_vindr_names(self, oracle):
        facts = oracle.get_facts(KNOWN_IMAGE_ID)
        for f in facts:
            assert f.concept in VINDR_NAMES

    def test_multi_rad_merged(self, oracle):
        # Oracle should have fewer facts per image than raw rows
        facts = oracle.get_facts(KNOWN_IMAGE_ID)
        concepts = [f.concept for f in facts]
        assert len(concepts) == len(set(concepts)), "Duplicate concepts — merge failed"

    def test_vqa_gt_finding_coverage(self, oracle):
        import json
        with open("data/vindr_cxr_vqa/vqa.json") as f:
            vqa = json.load(f)

        missing = []
        for img in vqa:
            img_id = img["image_id"]
            facts = oracle.get_facts(img_id)
            fact_concepts = {f.concept for f in facts}
            for item in img["vqa"]:
                gt = item["gt_finding"]
                if gt not in fact_concepts:
                    missing.append((img_id[:8], gt))

        assert len(missing) == 0, f"{len(missing)} missing gt_findings, e.g.: {missing[:5]}"
