"""Tests for SLAKE VQA evaluation pipeline.

All tests are CPU-only (mocked models, mocked APIs).
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# ── Task 1: SlakeKG tests ──

from src.knowledge.slake_kg import SlakeKG


@pytest.fixture
def mini_kg(tmp_path):
    """Create a minimal SLAKE KG on disk."""
    kg_dir = tmp_path / "KG"
    kg_dir.mkdir()
    (kg_dir / "en_disease.csv").write_text(
        "disease#attribute#value\n"
        "Pneumonia#cause#bacterial infection\n"
        "Pneumonia#symptom#cough, fever\n"
        "Pneumonia#location#Lung\n"
        "Pneumonia#treatment#antibiotics\n",
        encoding="utf-8",
    )
    (kg_dir / "en_organ.csv").write_text(
        "organ#attribute#value\n"
        "Lung#function#Breathe\n"
        "Heart#function#Pump blood\n",
        encoding="utf-8",
    )
    (kg_dir / "en_organ_rel.csv").write_text(
        "organ#attribute#value\n"
        "Heart#belong to#Circulatory System\n"
        "Lung#belong to#Respiratory System\n",
        encoding="utf-8",
    )
    return SlakeKG(str(kg_dir))


def test_slake_kg_lookup(mini_kg):
    assert mini_kg.lookup("Pneumonia", "cause") == "bacterial infection"
    assert mini_kg.lookup("Pneumonia", "symptom") == "cough, fever"
    assert mini_kg.lookup("Lung", "function") == "Breathe"
    assert mini_kg.lookup("Heart", "belong to") == "Circulatory System"


def test_slake_kg_case_insensitive(mini_kg):
    assert mini_kg.lookup("pneumonia", "CAUSE") == "bacterial infection"
    assert mini_kg.lookup("LUNG", "Function") == "Breathe"


def test_slake_kg_missing(mini_kg):
    assert mini_kg.lookup("Unknown", "cause") is None
    assert mini_kg.lookup("Pneumonia", "nonexistent") is None


def test_slake_kg_diseases(mini_kg):
    diseases = mini_kg.diseases()
    assert "pneumonia" in diseases


def test_slake_kg_organs(mini_kg):
    organs = mini_kg.organs()
    assert "lung" in organs
    assert "heart" in organs
