"""Per-system predictors for the multi-hop shared-cause QA.

Each predictor maps a QA item to a prediction {answer, cause, trace} in the
grading schema. Model-based predictors live in Phase B; predict_mock here is a
deterministic KG oracle used to validate the predict -> grade pipeline without a
model (a test fixture, NOT a reported system).
"""
import json
import re


def build_sc_prompt(item, mode):
    """Prompt for a shared-cause question. mode: zero_shot | cot | react."""
    q = item["question"]
    if mode == "cot":
        return (q + "\nReason step by step about conditions that could cause both "
                "findings, then end with a final line: 'Answer: Yes, <condition>' "
                "or 'Answer: No'.")
    return (q + "\nReply with 'Yes, <condition>' if a single condition can cause both, "
            "otherwise 'No'.")


def parse_yes_no_cause(text):
    """Parse model text -> (answer, cause). Safe default ('No', None) on junk."""
    if not text or not isinstance(text, str):
        return ("No", None)
    match = re.search(r"answer\s*[:\-]\s*(.+)", text, re.I)
    segment = (match.group(1) if match else text).strip()
    if re.match(r"(?i)\s*yes\b", segment):
        cause = None
        cm = re.search(r"(?i)yes[\s,:\-]+(.+)", segment)
        if cm:
            cause = cm.group(1).splitlines()[0].strip().rstrip(".").strip() or None
        return ("Yes", cause)
    return ("No", None)


def predict_mock(item, dag):
    """KG oracle: answer straight from the shared-cause set (verified trace)."""
    causes = dag.common_causes(item["finding_a"], item["finding_b"])
    if not causes:
        return {"answer": "No", "cause": None, "trace": []}
    cause = causes[0]
    return {
        "answer": "Yes",
        "cause": cause,
        "trace": [[cause, item["finding_a"]], [cause, item["finding_b"]]],
    }


def write_predictions(items, predict_fn, path):
    """Run predict_fn(item) over items and write predictions JSONL (with id)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            pred = predict_fn(item)
            f.write(json.dumps({"id": item["id"], **pred}) + "\n")
