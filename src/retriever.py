"""Búsqueda vectorial de fragmentos relevantes."""
from dataclasses import dataclass

from src.indexer import Indexer


@dataclass
class Hit:
    content: str
    title: str
    path: str
    section: str
    url: str
    score: float      # similitud coseno (1 = idéntico)
    page_id: str
    id: str = ""
    similarity: float | None = None   # similitud coseno (solo búsqueda vectorial)


class Retriever:
    def __init__(self, indexer: Indexer, top_k: int = 5):
        self.indexer = indexer
        self.top_k = top_k

    @staticmethod
    def _where(page_id):
        if not page_id:
            return None
        if isinstance(page_id, str):
            return {"page_id": page_id}
        return {"page_id": {"$in": list(page_id)}}

    def search(self, query: str, k: int | None = None,
               page_id: str | list[str] | None = None) -> list[Hit]:
        total = self.indexer.count()
        if total == 0:
            return []
        vec = self.indexer.embedder.embed([query])[0]
        res = self.indexer.col.query(
            query_embeddings=[vec], n_results=min(k or self.top_k, total),
            where=self._where(page_id),
            include=["documents", "metadatas", "distances"])
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            sim = round(1 - dist, 3)
            hits.append(Hit(doc, meta["title"], meta["path"], meta["section"],
                            meta["url"], sim, meta["page_id"], cid, sim))
        return hits
