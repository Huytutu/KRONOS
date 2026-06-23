import yaml


LLM_LINK_PROMPT = """Map this medical finding to one of the canonical names below.
Return exactly the canonical name, nothing else.

Canonical names: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, Pulmonary fibrosis

Finding: "{text}"
"""


class ConceptLinker:
    """Maps a free-text finding name to its canonical DAG node name.

    Tier 1: exact lookup in a synonym dictionary built from synonyms.yaml.
    Tier 2: LLM fallback (if llm_client provided) — validated against canonical names.
    """

    def __init__(self, synonyms_path, llm_client=None):
        with open(synonyms_path, encoding="utf-8") as f:
            entries = yaml.safe_load(f)

        self.lookup = {}
        self.canonical_names = set()
        for entry in entries:
            canonical = entry["canonical"]
            self.canonical_names.add(canonical)
            for synonym in entry["synonyms"]:
                self.lookup[synonym.lower()] = canonical

        self.llm_client = llm_client

    def link(self, text):
        """Return the canonical name for a finding, or None if no match."""
        if not text:
            return None
        result = self.lookup.get(text.strip().lower())
        if result is not None:
            return result
        if self.llm_client:
            return self._llm_link(text)
        return None

    def link_batch(self, texts):
        return [self.link(t) for t in texts]

    def _llm_link(self, text):
        """Tier-2 fallback: ask LLM to map to canonical name. Returns str or None."""
        raw = self.llm_client(LLM_LINK_PROMPT.format(text=text))
        if not raw:
            return None
        candidate = raw.strip()
        canonical_lower = {n.lower(): n for n in self.canonical_names}
        return canonical_lower.get(candidate.lower())
