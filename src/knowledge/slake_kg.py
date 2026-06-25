"""SLAKE 1.0 knowledge graph — disease/organ attribute lookup.

Loads three #-separated CSVs:
  en_disease.csv:   disease # attribute # value
  en_organ.csv:     organ # attribute # value
  en_organ_rel.csv: organ # relation  # value
"""
from pathlib import Path


class SlakeKG:
    def __init__(self, kg_dir="data/Slake1.0/KG"):
        self._data = {}
        kg = Path(kg_dir)
        for csv_file in ["en_disease.csv", "en_organ.csv", "en_organ_rel.csv"]:
            path = kg / csv_file
            if path.exists():
                self._load_csv(path)

    def _load_csv(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("disease#") or line.startswith("organ#"):
                    continue
                parts = line.split("#", 2)
                if len(parts) != 3:
                    continue
                entity, attr, value = parts
                key = entity.strip().lower()
                self._data.setdefault(key, {})[attr.strip().lower()] = value.strip()

    def lookup(self, entity, relation):
        entry = self._data.get(entity.strip().lower())
        if entry is None:
            return None
        return entry.get(relation.strip().lower())

    def diseases(self):
        return [k for k, v in self._data.items() if "symptom" in v or "cause" in v]

    def organs(self):
        return [k for k, v in self._data.items() if "function" in v or "belong to" in v]
