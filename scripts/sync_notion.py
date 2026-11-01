"""Descarga tus páginas de Notion a Markdown (data/raw) y sus imágenes (data/images).

Uso:
    python3 -m scripts.sync_notion          # solo lo que cambió
    python3 -m scripts.sync_notion --full   # reescribe todo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape

from config import settings
from src.notion_loader import NotionError
from src.pipeline import run_sync

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="ignora el estado y reescribe todo")
    args = parser.parse_args()

    errors = settings.validate_notion()
    if errors:
        for e in errors:
            console.print(f"[red]✗ {e}[/red]")
        return 1

    console.print("[bold]Sincronizando con Notion...[/bold]")
    try:
        result = run_sync(settings, full=args.full,
                          progress=lambda m: console.print(f"  • {escape(m)}"))
    except NotionError as e:
        console.print(f"[red]✗ {e}[/red]")
        return 1

    console.print(f"\n[bold green]Listo:[/bold green] {result.total} páginas, "
                  f"{result.changed} nuevas/actualizadas, {result.deleted} eliminadas, "
                  f"{result.images} imágenes descargadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
