"""Prueba la recuperación híbrida sin LLM:  python3 -m scripts.search "tu pregunta" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape

from config import settings
from src.embeddings import EmbeddingError, OllamaEmbedder
from src.hybrid_retriever import HybridRetriever
from src.indexer import Indexer, IndexerError

console = Console()


def main() -> int:
    if len(sys.argv) < 2:
        console.print('Uso: python3 -m scripts.search "tu pregunta"')
        return 1
    query = " ".join(sys.argv[1:])
    try:
        indexer = Indexer(settings.vectorstore_dir,
                          OllamaEmbedder(settings.ollama_host, settings.embedding_model))
        hits = HybridRetriever(indexer, settings.top_k).search(query)
    except (EmbeddingError, IndexerError) as e:
        console.print(f"[red]✗ {e}[/red]")
        return 1
    if not hits:
        console.print("[yellow]Índice vacío. Ejecuta sync_notion e index_notion.[/yellow]")
        return 1
    for i, h in enumerate(hits, 1):
        where = f"{h.path} > {h.section}" if h.section else h.path
        sim = f" · similitud {h.similarity}" if h.similarity is not None else " · solo coincidencia de palabras"
        console.print(f"\n[bold]{i}. {escape(where)}[/bold]  [cyan]relevancia {h.score}{sim}[/cyan]")
        snippet = h.content.replace("\n", " ")
        console.print(f"   {snippet[:300]}{'…' if len(snippet) > 300 else ''}", markup=False)
        console.print(f"   [dim]{h.url}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
