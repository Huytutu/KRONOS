from ultralytics import YOLO
from src.contracts import PerceptualFact


class Detector:
    """YOLOv12s wrapper — runs inference on CXR images, returns PerceptualFact list."""

    def __init__(self, weights_path, dag=None, conf_threshold=0.3):
        self.model = YOLO(weights_path)
        self.dag = dag
        self.conf_threshold = conf_threshold

    def detect(self, image_path):
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        return self._results_to_facts(results)

    def detect_batch(self, image_paths):
        return [self.detect(p) for p in image_paths]

    def _results_to_facts(self, results):
        if not results or results[0].boxes is None:
            return []

        r = results[0]
        img_w = r.orig_shape[1]
        img_h = r.orig_shape[0]
        facts = []

        for box, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            name = self.model.names[int(cls_id)]
            bbox = tuple(float(v) for v in box)
            lat = self._get_laterality(bbox, img_w, img_h)

            facts.append(PerceptualFact(
                concept=name,
                bbox=bbox,
                laterality=lat,
                conf=float(conf),
            ))

        return facts

    def _get_laterality(self, bbox, img_w, img_h):
        if self.dag:
            return self.dag.compose_laterality(bbox, img_w, img_h)

        center_x = (bbox[0] + bbox[2]) / 2 / img_w
        width_ratio = (bbox[2] - bbox[0]) / img_w

        if width_ratio > 0.4:
            return "bilateral"
        if abs(center_x - 0.5) < 0.12 and width_ratio < 0.35:
            return "midline"
        if center_x < 0.5:
            return "right"
        return "left"
