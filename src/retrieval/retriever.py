"""Retriever — wraps an index with a cached query embedding."""


class Retriever:
    def __init__(self, index, encoder=None):
        self.index = index
        self.encoder = encoder
        self.query_emb = None

    def set_query_emb(self, emb):
        self.query_emb = emb

    def retrieve(self, k=5):
        if self.query_emb is None:
            return []
        results = self.index.search(self.query_emb, k=k)
        return [{"score": score, **case} for score, case in results]
