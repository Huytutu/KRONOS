import csv
import json
from collections import defaultdict
from pathlib import Path
from src.contracts import PerceptualFact


class PerceptionOracle:
    """Ground-truth perception from train.csv — for evaluation and testing."""

    def __init__(self, train_csv_path, dag=None):
        self.dag = dag
        self._facts = {}  # image_id -> list[PerceptualFact]
        self._all_image_ids = set()
        self._load(train_csv_path)

    def get_facts(self, image_id):
        return self._facts.get(image_id, [])

    def get_all_image_ids(self):
        return self._all_image_ids

    def _load(self, path):
        # Group rows by (image_id, class_name), collect bboxes + image size
        grouped = defaultdict(lambda: {"bboxes": [], "width": 0, "height": 0})

        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row["x_min"]:
                    continue

                key = (row["image_id"], row["class_name"])
                grouped[key]["bboxes"].append((
                    float(row["x_min"]), float(row["y_min"]),
                    float(row["x_max"]), float(row["y_max"]),
                ))
                grouped[key]["width"] = int(row["width"])
                grouped[key]["height"] = int(row["height"])

        # Merge multi-rad bboxes into one fact per (image, finding)
        image_facts = defaultdict(list)
        for (image_id, class_name), data in grouped.items():
            self._all_image_ids.add(image_id)
            bboxes = data["bboxes"]
            orig_w = data["width"]
            orig_h = data["height"]

            # Average bbox across radiologists, scaled to 512x512
            scale_x = 512.0 / orig_w
            scale_y = 512.0 / orig_h
            avg_bbox = (
                sum(b[0] for b in bboxes) / len(bboxes) * scale_x,
                sum(b[1] for b in bboxes) / len(bboxes) * scale_y,
                sum(b[2] for b in bboxes) / len(bboxes) * scale_x,
                sum(b[3] for b in bboxes) / len(bboxes) * scale_y,
            )

            lat = self._get_laterality(avg_bbox)

            image_facts[image_id].append(PerceptualFact(
                concept=class_name,
                bbox=avg_bbox,
                laterality=lat,
                conf=1.0,
            ))

        self._facts = dict(image_facts)

    def _get_laterality(self, bbox):
        if self.dag:
            return self.dag.compose_laterality(bbox, 512, 512)

        center_x = (bbox[0] + bbox[2]) / 2 / 512
        width_ratio = (bbox[2] - bbox[0]) / 512

        if width_ratio > 0.4:
            return "bilateral"
        if abs(center_x - 0.5) < 0.12 and width_ratio < 0.35:
            return "midline"
        if center_x < 0.5:
            return "right"
        return "left"


class SlakeOracle:
    """Ground-truth detector for SLAKE — reads detection.json per image.

    SLAKE detection format: [{"Finding": [x, y, w, h]}, ...]
    Acts as a drop-in for Detector: call detect(image_path) → List[PerceptualFact].
    """

    def __init__(self, image_dir="data/Slake1.0/imgs"):
        self.image_dir = Path(image_dir)

    def detect(self, image_path):
        img_dir = Path(image_path).parent
        det_path = img_dir / "detection.json"
        if not det_path.exists():
            return []
        with open(det_path, encoding="utf-8") as f:
            entries = json.load(f)
        facts = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for finding, coords in entry.items():
                if not finding or not coords or len(coords) != 4:
                    continue
                x, y, w, h = coords
                bbox = (x, y, x + w, y + h)
                facts.append(PerceptualFact(
                    concept=finding, bbox=bbox, laterality="midline", conf=1.0,
                ))
        return facts
