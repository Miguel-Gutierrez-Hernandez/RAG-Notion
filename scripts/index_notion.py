"""Indexa en ChromaDB las páginas pendientes (nuevas, modificadas o eliminadas).

Uso:
    python3 -m scripts.index_notion            # solo lo pendiente
    python3 -m scripts.index_notion --rebuild  # reindexa todo desde cero
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from config import settings
from src.embeddings import EmbeddingError, OllamaEmbedder
from src.indexer import IndexerError, Indexer
from src.sync_state import load_state, save_state

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="borra el índice y reindexa todo")
    args = parser.parse_args()

    embedder = OllamaEmbedder(settings.ollama_host, settings.embedding_model)
    try:
        embedder.embed(["ping"])  # comprueba Ollama y el modelo antes de empezar
        indexer = Indexer(settings.vectorstore_dir, embedder, settings.chunk_size,
                          settings.chunk_overlap, rebuild=args.rebuild)
    except (EmbeddingError, IndexerError) as e:
        console.print(f"[red]✗ {e}[/red]")
        return 1

    state = load_state(settings.sync_state_file)
    pending = state["pending"]
    if args.rebuild:
        pending["changed"], pending["deleted"] = list(state["pages"]), []
        save_state(settings.sync_state_file, state)

    if not pending["changed"] and not pending["deleted"]:
        console.print("[green]Nada pendiente.[/green] Ejecuta primero scripts.sync_notion si editaste Notion.")
        console.print(f"Fragmentos en el índice: {indexer.count()}")
        return 0

    for pid in list(pending["deleted"]):
        indexer.delete_page(pid)
        pending["deleted"].remove(pid)
        save_state(settings.sync_state_file, state)
        console.print(f"  • {pid}  [red]eliminada del índice[/red]")

    todo = list(pending["changed"])
    for n, pid in enumerate(todo, 1):
        file = settings.raw_dir / f"{pid}.md"
        label = state["pages"].get(pid, {}).get("path", pid)
        if not file.exists():
            console.print(f"  [yellow]! {label}: falta {file.name}, ejecuta sync_notion[/yellow]")
            pending["changed"].remove(pid)
            continue
        try:
            n_chunks = indexer.index_page(file.read_text(encoding="utf-8"))
        except EmbeddingError as e:
            console.print(f"[red]✗ {e}[/red]\nLo indexado hasta aquí se conserva; vuelve a ejecutar para continuar.")
            return 1
        pending["changed"].remove(pid)
        save_state(settings.sync_state_file, state)  # reanudable si se interrumpe
        console.print(f"  [{n}/{len(todo)}] {label}  [dim]{n_chunks} fragmentos[/dim]")

    save_state(settings.sync_state_file, state)
    console.print(f"\n[bold green]Listo.[/bold green] Fragmentos en el índice: {indexer.count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
