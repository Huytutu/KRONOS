import yaml


class ConceptLinker:
    """Maps a free-text finding name to its canonical DAG node name.

    Tier 1 only: exact lookup in a synonym dictionary built from synonyms.yaml.
    Unknown text returns None — never a guess. Tier 2 (fuzzy/LLM) is future work.
    """

    def __init__(self, synonyms_path):
        with open(synonyms_path, encoding="utf-8") as f:
            entries = yaml.safe_load(f)

        # One flat dict: lowercase synonym -> canonical name.
        self.lookup = {}
        for entry in entries:
            canonical = entry["canonical"]
            for synonym in entry["synonyms"]:
                self.lookup[synonym.lower()] = canonical

    def link(self, text):
        """Return the canonical name for a finding, or None if no match."""
        if not text:
            return None
        return self.lookup.get(text.strip().lower())

    def link_batch(self, texts):
        return [self.link(t) for t in texts]
