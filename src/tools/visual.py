"""Visual tools — inspect, re_detect, compare.

Each takes an Action + image + injectable callable (VLM or detector) → Observation.
Results are meant to be folded into the evidence graph as new PerceptualFact entries.
"""
import json
from src.contracts import Action, Observation, PerceptualFact


def _parse_bbox(raw):
    """Extract (x1, y1, x2, y2) floats from LLM-generated bbox arg.
    Returns a 4-tuple of floats, or None if unparseable."""
    if isinstance(raw, (list, tuple)):
        nums = raw
    elif isinstance(raw, str):
        cleaned = raw.strip("()[] ")
        nums = cleaned.split(",")
    else:
        return None
    try:
        vals = [float(v) for v in nums[:4]]
    except (ValueError, TypeError):
        return None
    if len(vals) != 4:
        return None
    return tuple(vals)


def run_inspect(action, image, vlm_fn):
    if image is None:
        return Observation(result=None, ok=False)

    bbox = _parse_bbox(action.args.get("bbox"))
    if bbox is None:
        return Observation(result=None, ok=False)
    crop = _crop(image, bbox)
    prompt = "Describe the abnormality in this chest X-ray region. Return JSON: {\"concept\": \"<finding>\", \"conf\": <0-1>, \"description\": \"<text>\"}"

    try:
        raw = vlm_fn(crop, prompt)
        parsed = json.loads(raw)
        if "concept" in parsed:
            return Observation(result=parsed, ok=True)
        return Observation(result=None, ok=False)
    except (json.JSONDecodeError, TypeError, KeyError):
        return Observation(result=None, ok=False)


def run_re_detect(action, image, detector_fn):
    if image is None:
        return Observation(result=None, ok=False)

    bbox = _parse_bbox(action.args.get("bbox"))
    if bbox is None:
        return Observation(result=None, ok=False)
    x1, y1, x2, y2 = bbox
    crop = _crop(image, bbox)

    raw_facts = detector_fn(crop)
    if not raw_facts:
        return Observation(result=None, ok=False)

    mapped = []
    for f in raw_facts:
        fx1, fy1, fx2, fy2 = f.bbox
        mapped.append(PerceptualFact(
            concept=f.concept,
            bbox=(fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1),
            laterality=f.laterality,
            conf=f.conf,
        ))

    return Observation(result=mapped, ok=True)


def run_compare(action, image, vlm_fn):
    if image is None:
        return Observation(result=None, ok=False)

    bbox1 = _parse_bbox(action.args.get("bbox1"))
    bbox2 = _parse_bbox(action.args.get("bbox2"))
    if bbox1 is None or bbox2 is None:
        return Observation(result=None, ok=False)
    crop1 = _crop(image, bbox1)
    crop2 = _crop(image, bbox2)
    prompt = "Compare these two chest X-ray regions. Return JSON: {\"comparison\": \"<text>\", \"laterality_hint\": \"left|right|bilateral\"}"

    try:
        raw = vlm_fn(crop1, crop2, prompt)
        parsed = json.loads(raw)
        if "comparison" in parsed:
            return Observation(result=parsed, ok=True)
        return Observation(result=None, ok=False)
    except (json.JSONDecodeError, TypeError, KeyError):
        return Observation(result=None, ok=False)


def fold_facts(existing, new_facts, iou_threshold=0.5):
    merged = list(existing)
    for nf in new_facts:
        is_dup = False
        for ef in merged:
            if ef.concept == nf.concept and _iou(ef.bbox, nf.bbox) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            merged.append(nf)
    return merged


def _crop(image, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = image.size
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
