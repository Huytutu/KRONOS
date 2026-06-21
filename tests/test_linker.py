"""Unit tests for ConceptLinker: free-text finding name -> canonical DAG node."""

import pytest
from src.linking.linker import ConceptLinker

SYNONYMS_PATH = "data/ontology/synonyms.yaml"

# The 14 VinDr canonical names. Each must link to itself.
VINDR_CANONICALS = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]


@pytest.fixture(scope="module")
def linker():
    return ConceptLinker(SYNONYMS_PATH)


@pytest.mark.parametrize("name", VINDR_CANONICALS)
def test_canonical_links_to_itself(linker, name):
    assert linker.link(name) == name


@pytest.mark.parametrize("variant,expected", [
    ("enlarged heart", "Cardiomegaly"),
    ("cardiac enlargement", "Cardiomegaly"),
    ("lung collapse", "Atelectasis"),
    ("ground glass opacity", "Lung Opacity"),
    ("ggo", "Lung Opacity"),
    ("ipf", "Pulmonary fibrosis"),
    ("hydrothorax", "Pleural effusion"),
    ("tension pneumothorax", "Pneumothorax"),
    ("interstitial lung disease", "ILD"),
    ("infiltrate", "Infiltration"),
])
def test_synonym_variant_resolves(linker, variant, expected):
    assert linker.link(variant) == expected


def test_case_insensitive(linker):
    assert linker.link("PNEUMOTHORAX") == "Pneumothorax"
    assert linker.link("CardioMegaly") == "Cardiomegaly"


def test_surrounding_whitespace_ignored(linker):
    assert linker.link("  cardiomegaly  ") == "Cardiomegaly"


def test_unknown_returns_none(linker):
    assert linker.link("xyzzy nonsense") is None


def test_empty_string_returns_none(linker):
    assert linker.link("") is None


def test_link_batch(linker):
    result = linker.link_batch(["cardiomegaly", "xyzzy", "pneumothorax"])
    assert result == ["Cardiomegaly", None, "Pneumothorax"]
