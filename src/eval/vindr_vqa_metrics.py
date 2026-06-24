"""Grading logic for VinDr-CXR VQA evaluation.

Uses an LLM judge (Gemini) to score free-text predictions against ground truth.
Reports accuracy overall, by question type, and by difficulty.
"""

JUDGE_PROMPT = """You are a medical imaging expert grading a VQA system's answer.

Question: {question}
Ground truth answer: {ground_truth}
System prediction: {prediction}

Does the prediction correctly answer the question? Consider:
- For Yes/No questions: is the Yes/No part correct?
- For location questions: does it identify the correct region?
- For counting questions: is the count correct?
- For identification questions: does it name the correct finding(s)?

Reply with exactly "CORRECT" or "INCORRECT"."""


def judge_answer(question, prediction, ground_truth, llm_fn):
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    resp = llm_fn(prompt)
    return 1 if "CORRECT" in resp.upper() and "INCORRECT" not in resp.upper() else 0


def grade_batch(items, predictions, llm_fn):
    # Smoke-test the judge before grading the full batch
    test_resp = llm_fn("Reply with exactly CORRECT")
    if not test_resp or "CORRECT" not in test_resp.upper():
        import warnings
        warnings.warn(
            "LLM judge returned empty/invalid response — is GEMINI_API_KEY set? "
            "All scores will be 0.", stacklevel=2,
        )

    scores = []
    for item, pred in zip(items, predictions):
        score = judge_answer(item.question, pred, item.answer, llm_fn)
        scores.append((item, score))

    n = len(scores)
    overall = sum(s for _, s in scores) / n if n else 0.0

    by_type = {}
    by_difficulty = {}
    for item, score in scores:
        qtype = item.meta.get("type", "unknown")
        diff = item.meta.get("difficulty", "unknown")

        by_type.setdefault(qtype, []).append(score)
        by_difficulty.setdefault(diff, []).append(score)

    def _agg(group):
        return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in group.items()}

    return {
        "n": n,
        "overall_accuracy": overall,
        "by_type": _agg(by_type),
        "by_difficulty": _agg(by_difficulty),
        "details": [
            {"id": item.id, "question": item.question, "prediction": pred,
             "ground_truth": item.answer, "score": score}
            for (item, score), pred in zip(scores, predictions)
        ],
    }
