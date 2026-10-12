"""Cliente del modelo de chat local (Ollama) con streaming."""
import json
from typing import Iterator

import requests


class LLMError(Exception):
    pass


class OllamaChat:
    def __init__(self, host: str, model: str, temperature: float = 0.2,
                 num_ctx: int = 4096):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx   # Ollama usa 2048 por defecto y recortaría el contexto

    def stream(self, messages: list[dict], **options) -> Iterator[str]:
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx, **options}
        try:
            r = requests.post(f"{self.host}/api/chat", stream=True, timeout=(10, 600),
                              json={"model": self.model, "messages": messages,
                                    "stream": True, "options": opts})
        except requests.ConnectionError:
            raise LLMError(f"Ollama no responde en {self.host}. "
                           "Inícialo con: brew services start ollama")
        if r.status_code == 404:
            raise LLMError(f"Modelo '{self.model}' no encontrado. "
                           f"Descárgalo con: ollama pull {self.model}")
        if r.status_code != 200:
            raise LLMError(f"Error de Ollama ({r.status_code}): {r.text[:200]}")
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise LLMError(data["error"])
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break
