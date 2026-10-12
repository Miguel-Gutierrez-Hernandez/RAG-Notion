"""Base vectorial local (ChromaDB): indexa, actualiza y borra páginas."""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunker import chunk_page

COLLECTION = "notion"


class IndexerError(Exception):
    pass


class Indexer:
    def __init__(self, store_dir: Path, embedder, chunk_size: int = 800,
                 chunk_overlap: int = 100, rebuild: bool = False):
        Path(store_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(store_dir), settings=ChromaSettings(anonymized_telemetry=False))
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if rebuild:
            self.col = None
            self.reset()
        else:
            self.col = self._open()

    def _exists(self) -> bool:
        return COLLECTION in {getattr(c, "name", c) for c in self.client.list_collections()}

    def _create(self):
        return self.client.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine",
                                  "embedding_model": self.embedder.model})

    def _open(self):
        if not self._exists():
            return self._create()
        col = self.client.get_collection(COLLECTION)
        used = (col.metadata or {}).get("embedding_model")
        if used and used != self.embedder.model:
            raise IndexerError(
                f"El índice se creó con '{used}' y ahora usas '{self.embedder.model}'. "
                "Ejecuta con --rebuild para reindexar todo.")
        return col

    def reset(self) -> None:
        if self._exists():
            self.client.delete_collection(COLLECTION)
        self.col = self._create()

    def index_page(self, md_text: str) -> int:
        chunks = chunk_page(md_text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        # Primero embeddings; si fallan, el índice anterior de la página queda intacto.
        vectors = self.embedder.embed([c.embed_text for c in chunks])
        self.delete_page(chunks[0].page_id)
        self.col.add(ids=[c.id for c in chunks], embeddings=vectors,
                     documents=[c.content for c in chunks],
                     metadatas=[c.metadata for c in chunks])
        return len(chunks)

    def delete_page(self, page_id: str) -> None:
        self.col.delete(where={"page_id": page_id})

    def count(self) -> int:
        return self.col.count()
