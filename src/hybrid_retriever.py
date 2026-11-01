"""Búsqueda híbrida: vectorial (significado) + BM25 (palabras exactas), fusionadas con RRF."""
import math
import re
import unicodedata
from dataclasses import replace

from src.indexer import Indexer
from src.retriever import Hit, Retriever, parse_images

STOPWORDS = set("""
a al algo ante con contra cual cuando de del desde donde e el ella ellos en entre era es esa ese eso esta
este esto estoy fue ha han hay la las le les lo los me mi mis mas muy ni no nos o para pero por que quien
se si sin sobre son su sus te ti tu tus un una uno unos unas y ya yo como cuanto cuantos cuantas cual cuales
""".split())


def _strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _stem(token: str) -> str:
    """Normalización muy ligera de plurales (gatos/gato, ciudades/ciudad)."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", _strip_accents(text))
    return [_stem(t) for t in tokens if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(docs):
            for term in doc:
                self.postings.setdefault(term, {})
                self.postings[term][i] = self.postings[term].get(i, 0) + 1

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = {}
        for term in set(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1))
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / norm
        return out


class HybridRetriever:
    def __init__(self, indexer: Indexer, top_k: int = 5, candidates: int = 20,
                 rrf_k: int = 60):
        self.indexer = indexer
        self.top_k = top_k
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.vector = Retriever(indexer, top_k)
        self._fingerprint = None
        self._bm25: BM25 | None = None
        self._rows: list[tuple[str, str, dict]] = []   # (id, contenido, metadatos)

    def _ensure_index(self) -> None:
        """Reconstruye el índice de palabras si el índice vectorial cambió."""
        metas = self.indexer.col.get(include=["metadatas"])
        fp = hash(tuple(sorted((i, m.get("last_edited", "")) for i, m in
                               zip(metas["ids"], metas["metadatas"]))))
        if fp == self._fingerprint:
            return
        data = self.indexer.col.get(include=["documents", "metadatas"])
        self._rows = list(zip(data["ids"], data["documents"], data["metadatas"]))
        self._bm25 = BM25([tokenize(f"{m['path']} {m['section']} {d}")
                           for _, d, m in self._rows])
        self._fingerprint = fp

    def _keyword_hits(self, query: str, page_id: str | list[str] | None,
                      n: int) -> list[Hit]:
        self._ensure_index()
        allowed = None if not page_id else ({page_id} if isinstance(page_id, str)
                                            else set(page_id))
        scores = self._bm25.scores(tokenize(query))
        ranked = sorted(scores, key=scores.get, reverse=True)
        hits = []
        for i in ranked:
            cid, doc, m = self._rows[i]
            if allowed is not None and m["page_id"] not in allowed:
                continue
            hits.append(Hit(doc, m["title"], m["path"], m["section"], m["url"],
                            0.0, m["page_id"], cid, images=parse_images(m)))
            if len(hits) >= n:
                break
        return hits

    def search(self, query: str, k: int | None = None,
               page_id: str | list[str] | None = None) -> list[Hit]:
        if self.indexer.count() == 0:
            return []
        k = k or self.top_k
        vec = self.vector.search(query, self.candidates, page_id)
        kw = self._keyword_hits(query, page_id, self.candidates)

        fused: dict[str, float] = {}
        by_id: dict[str, Hit] = {}
        for ranking in (vec, kw):
            for rank, hit in enumerate(ranking):
                fused[hit.id] = fused.get(hit.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if hit.id not in by_id or hit.similarity is not None:
                    by_id[hit.id] = hit
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [replace(by_id[i], score=round(fused[i], 4)) for i in top]
