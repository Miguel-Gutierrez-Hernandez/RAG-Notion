"""Configuración central del proyecto (lee variables desde .env)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # Notion
    notion_token: str = os.getenv("NOTION_TOKEN", "")
    notion_root_page_id: str = os.getenv("NOTION_ROOT_PAGE_ID", "")

    # Ollama / modelos
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    chat_model: str = os.getenv("CHAT_MODEL", "")  # vacío = automático

    # RAG
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 100))
    top_k: int = int(os.getenv("TOP_K", 5))

    # Rutas
    raw_dir: Path = BASE_DIR / "data" / "raw"
    images_dir: Path = BASE_DIR / "data" / "images"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"
    sync_state_file: Path = BASE_DIR / "data" / "sync_state.json"

    def validate_notion(self) -> list[str]:
        errors = []
        if not self.notion_token:
            errors.append("Falta NOTION_TOKEN en .env")
        if not self.notion_root_page_id:
            errors.append("Falta NOTION_ROOT_PAGE_ID en .env")
        return errors


settings = Settings()
