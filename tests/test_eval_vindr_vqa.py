"""Tests for VinDr-CXR VQA evaluation pipeline.

All tests are CPU-only (mocked models, mocked APIs).
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Task 1: Gemini client tests ──

def test_gemini_complete_returns_text():
    mock_resp = MagicMock()
    mock_resp.text = "Hello world"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}), \
         patch("google.genai.Client", return_value=mock_client):
        from src.llm.gemini_client import complete
        result = complete("Say hello")
    assert result == "Hello world"


def test_gemini_complete_missing_key_returns_empty():
    with patch.dict("os.environ", {}, clear=True):
        import importlib
        import src.llm.gemini_client as mod
        importlib.reload(mod)
        result = mod.complete("test")
    assert result == ""


# ── Task 2: VinDr VQA metrics tests ──

from src.eval.vindr_vqa_metrics import judge_answer, grade_batch
from src.data.loaders import QAItem


def test_judge_answer_correct():
    llm_fn = lambda prompt: "CORRECT"
    assert judge_answer("Is there X?", "Yes", "Yes, X is present.", llm_fn) == 1


def test_judge_answer_incorrect():
    llm_fn = lambda prompt: "INCORRECT"
    assert judge_answer("Is there X?", "No", "Yes, X is present.", llm_fn) == 0


def test_judge_answer_parse_robustness():
    llm_fn = lambda prompt: "CORRECT. The answer matches the ground truth."
    assert judge_answer("Q?", "A", "A", llm_fn) == 1


def test_judge_answer_incorrect_substring():
    """'INCORRECT' contains 'CORRECT' — must parse as 0."""
    llm_fn = lambda prompt: "INCORRECT"
    assert judge_answer("Q?", "A", "B", llm_fn) == 0


def _make_items():
    """6 items: one per question type, mixed difficulty."""
    types = ["Where", "Is_there", "How_many", "Yes_No", "Which", "What"]
    diffs = ["Easy", "Medium", "Easy", "Medium", "Easy", "Medium"]
    items = []
    for i, (t, d) in enumerate(zip(types, diffs)):
        items.append(QAItem(
            id=f"img_{i}", dataset="vindr_vqa", image=f"img_{i}.png",
            question=f"Q{i}?", answer=f"A{i}",
            meta={"type": t, "difficulty": d},
        ))
    return items


def test_grade_batch_aggregation():
    items = _make_items()
    predictions = [f"A{i}" for i in range(6)]
    # Judge: first 4 correct, last 2 incorrect
    call_count = [0]
    def mock_llm(prompt):
        idx = call_count[0]
        call_count[0] += 1
        return "CORRECT" if idx < 4 else "INCORRECT"

    result = grade_batch(items, predictions, mock_llm)

    assert result["n"] == 6
    assert abs(result["overall_accuracy"] - 4 / 6) < 1e-9
    assert result["by_type"]["Where"]["accuracy"] == 1.0
    assert result["by_type"]["What"]["accuracy"] == 0.0
    assert result["by_difficulty"]["Easy"]["n"] == 3
    assert result["by_difficulty"]["Medium"]["n"] == 3
