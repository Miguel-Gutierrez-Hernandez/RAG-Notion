"""Embeddings locales con Ollama."""
import requests


class EmbeddingError(Exception):
    pass


class OllamaEmbedder:
    def __init__(self, host: str, model: str, batch_size: int = 16):
        self.host = host.rstrip("/")
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors += self._post(texts[i:i + self.batch_size])
        return vectors

    def _post(self, batch: list[str]) -> list[list[float]]:
        try:
            r = requests.post(f"{self.host}/api/embed",
                              json={"model": self.model, "input": batch, "truncate": True},
                              timeout=300)
        except requests.ConnectionError:
            raise EmbeddingError(
                f"Ollama no responde en {self.host}. Inícialo con: brew services start ollama")
        if r.status_code == 404:
            raise EmbeddingError(
                f"Modelo '{self.model}' no encontrado. Descárgalo con: ollama pull {self.model}")
        if r.status_code != 200:
            raise EmbeddingError(f"Error de Ollama ({r.status_code}): {r.text[:200]}")
        return r.json()["embeddings"]
