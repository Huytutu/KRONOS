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
