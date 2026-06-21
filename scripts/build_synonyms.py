"""
Fetch synonyms from BioPortal (RADLEX + MESH + SNOMEDCT) for VinDr findings
and generate data/ontology/synonyms.yaml.

Usage:
    python scripts/build_synonyms.py

Requires BIO_PORTAL_API_KEY in .env file.
"""

import os, time, requests, yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BIO_PORTAL_API_KEY")
BASE = "https://data.bioontology.org"
HEADERS = {"Authorization": f"apikey token={API_KEY}"}

# 14 VinDr findings: canonical name, RadLex RID, and relevance keywords
# for filtering BioPortal noise. Only synonyms containing at least one
# keyword (case-insensitive) are kept from API results.
FINDINGS = [
    {
        "canonical": "Aortic enlargement",
        "rid": "RID34873",
        "keywords": ["aortic", "aorta"],
        "extras": ["enlarged aorta", "aortic dilatation", "dilated aorta", "aortic ectasia"],
    },
    {
        "canonical": "Atelectasis",
        "rid": "RID28493",
        "keywords": ["atelectasis", "lung collapse", "collapsed lung", "pulmonary collapse"],
        "extras": ["collapsed lung"],
    },
    {
        "canonical": "Calcification",
        "rid": "RID5196",
        "keywords": ["calcification", "calcified"],
        "extras": [],
    },
    {
        "canonical": "Cardiomegaly",
        "rid": "RID1392",
        "keywords": ["cardiomegaly", "heart", "cardiac"],
        "extras": ["cardiac enlargement"],
    },
    {
        "canonical": "Consolidation",
        "rid": "RID43255",
        "keywords": ["consolidation", "lung consolidation"],
        "extras": ["pulmonary consolidation", "airspace consolidation"],
    },
    {
        "canonical": "ILD",
        "rid": "RID4864",
        "keywords": ["interstitial lung", "interstitial pulmonary", "ild", "parenchymal lung"],
        "extras": ["interstitial lung disease", "interstitial disease",
                    "diffuse parenchymal lung disease"],
    },
    {
        "canonical": "Infiltration",
        "rid": "RID28825",
        "keywords": ["infiltrat"],
        "extras": ["pulmonary infiltrate", "lung infiltrate", "pulmonary infiltration"],
    },
    {
        "canonical": "Lung Opacity",
        "rid": "RID28530",
        "keywords": ["lung opacity", "pulmonary opacity", "ground glass"],
        "extras": ["opacity", "pulmonary opacity", "ground glass opacity",
                    "ground-glass opacity", "GGO"],
    },
    {
        "canonical": "Nodule/Mass",
        "rid": "RID3875",
        "keywords": ["nodule", "pulmonary nodule", "lung nodule", "lung mass"],
        "extras": ["lung nodule", "pulmonary nodule", "lung mass", "mass",
                    "pulmonary mass", "solitary pulmonary nodule"],
    },
    {
        "canonical": "Other lesion",
        "rid": None,
        "keywords": [],
        "extras": ["other lesion", "other abnormality", "other finding"],
    },
    {
        "canonical": "Pleural effusion",
        "rid": "RID34539",
        "keywords": ["pleural effusion", "effusion"],
        "extras": ["fluid in pleural space", "hydrothorax"],
    },
    {
        "canonical": "Pleural thickening",
        "rid": "RID34771",
        "keywords": ["pleural thick", "thickening of pleura", "pleural cuirasse"],
        "extras": ["thickened pleura"],
    },
    {
        "canonical": "Pneumothorax",
        "rid": "RID5352",
        "keywords": ["pneumothorax"],
        "extras": ["air in pleural space", "collapsed lung due to air leak"],
    },
    {
        "canonical": "Pulmonary fibrosis",
        "rid": "RID4737",
        "keywords": ["pulmonary fibrosis", "lung fibrosis", "fibrosing alveolitis",
                      "interstitial fibrosis"],
        "extras": ["lung fibrosis", "idiopathic pulmonary fibrosis", "IPF"],
    },
]

# Terms that indicate a synonym is a procedure, not a finding
EXCLUDE_PATTERNS = [
    "(procedure)", "(product)", "surgery", "surgical", "antineoplastic",
    "chemotherapy", "radiotherapy", "radiation therapy", "administration of",
    "family history", "fetal", "foetal", "congenital", "drug induced",
    "neonatorum", "prematurity", "perinatal", "(situation)",
]


def fetch_synonyms(query):
    r = requests.get(
        f"{BASE}/search",
        params={
            "q": query,
            "ontologies": "RADLEX,MESH,SNOMEDCT",
            "include": "prefLabel,synonym",
            "pagesize": 10,
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    syns = set()
    for item in r.json().get("collection", []):
        pref = item.get("prefLabel", "")
        if pref:
            syns.add(pref.lower())
        for s in item.get("synonym", []):
            syns.add(s.lower())
    return syns


def is_relevant(synonym, keywords):
    s = synonym.lower()
    for excl in EXCLUDE_PATTERNS:
        if excl in s:
            return False
    for kw in keywords:
        if kw.lower() in s:
            return True
    return False


def build_entry(finding):
    canonical = finding["canonical"]
    keywords = finding["keywords"]

    # "Other lesion" has no BioPortal concept
    if not keywords:
        all_syns = set()
    else:
        api_syns = fetch_synonyms(canonical)
        all_syns = {s for s in api_syns if is_relevant(s, keywords)}
        time.sleep(0.3)

    # Add canonical name and hand-curated extras
    all_syns.add(canonical.lower())
    for extra in finding["extras"]:
        all_syns.add(extra.lower())

    entry = {"canonical": canonical, "synonyms": sorted(all_syns)}
    if finding["rid"]:
        entry["rid"] = finding["rid"]
    return entry


def main():
    print(f"Fetching synonyms from BioPortal...")
    entries = []
    for f in FINDINGS:
        entry = build_entry(f)
        entries.append(entry)
        print(f"  {entry['canonical']:25s} -> {len(entry['synonyms'])} synonyms")

    out_path = Path(__file__).parent.parent / "data" / "ontology" / "synonyms.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# Synonym dictionary for VinDr-CXR findings.\n")
        fh.write("# Source: BioPortal (RADLEX + MESH + SNOMEDCT) + hand-curated.\n")
        fh.write("# Used by ConceptLinker to map free-text -> canonical DAG node.\n\n")
        yaml.dump(entries, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
