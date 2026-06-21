"""Vector index for RAG retrieval — BruteForceIndex (numpy) and RagIndex (faiss)."""
import numpy as np


class BruteForceIndex:
    """Exact cosine search using numpy. No faiss dependency — for tests and small corpora."""

    def __init__(self, embeddings, cases):
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.cases = list(cases)

    def search(self, query_emb, k=5):
        query = np.array(query_emb, dtype=np.float32).ravel()
        scores = self.embeddings @ query
        k = min(k, len(self.cases))
        top_k = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.cases[i]) for i in top_k]


class RagIndex:
    """FAISS-backed exact inner-product index. Requires faiss-cpu."""

    def __init__(self, index, cases):
        self._index = index
        self.cases = list(cases)

    @classmethod
    def from_data(cls, embeddings, cases):
        import faiss
        emb = np.array(embeddings, dtype=np.float32)
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        return cls(index, cases)

    @classmethod
    def load(cls, faiss_path, cases):
        import faiss
        index = faiss.read_index(str(faiss_path))
        return cls(index, cases)

    def search(self, query_emb, k=5):
        query = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self.cases))
        scores, indices = self._index.search(query, k)
        return [(float(scores[0][i]), self.cases[indices[0][i]]) for i in range(k)]
