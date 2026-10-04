"""Descarga tus páginas de Notion a Markdown (data/raw) de forma incremental.

Uso:
    python3 -m scripts.sync_notion          # solo lo que cambió
    python3 -m scripts.sync_notion --full   # reescribe todo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from config import settings
from src.notion_loader import NotionClient, NotionError, crawl
from src.sync_state import load_state, mark_pending, save_state

console = Console()


def page_file(page_id: str) -> Path:
    return settings.raw_dir / f"{page_id}.md"


def to_markdown_file(page) -> str:
    header = (f"---\nid: {page.id}\ntitle: {page.title}\npath: {page.path_str}\n"
              f"url: {page.url}\nlast_edited: {page.last_edited}\n---\n\n")
    return f"{header}# {page.title}\n\n{page.markdown}\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="ignora el estado y reescribe todo")
    args = parser.parse_args()

    errors = settings.validate_notion()
    if errors:
        for e in errors:
            console.print(f"[red]✗ {e}[/red]")
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(settings.sync_state_file)
    old_pages = state["pages"]
    new_pages, changed = {}, []

    client = NotionClient(settings.notion_token)
    console.print("[bold]Sincronizando con Notion...[/bold]")
    try:
        for page in crawl(client, settings.notion_root_page_id):
            prev = old_pages.get(page.id)
            is_new = prev is None
            modified = is_new or prev["last_edited"] != page.last_edited \
                or not page_file(page.id).exists() or args.full
            if modified:
                page_file(page.id).write_text(to_markdown_file(page), encoding="utf-8")
                changed.append(page.id)
            new_pages[page.id] = {"title": page.title, "path": page.path_str,
                                  "url": page.url, "last_edited": page.last_edited}
            tag = ("[green]nueva[/green]" if is_new else
                   "[yellow]actualizada[/yellow]" if modified else "[dim]sin cambios[/dim]")
            console.print(f"  • {page.path_str}  {tag}")
    except NotionError as e:
        # No se guarda nada: así un fallo a mitad de camino nunca borra páginas.
        console.print(f"[red]✗ {e}[/red]")
        return 1

    deleted = [pid for pid in old_pages if pid not in new_pages]
    for pid in deleted:
        page_file(pid).unlink(missing_ok=True)
        console.print(f"  • {old_pages[pid]['path']}  [red]eliminada[/red]")

    state["pages"] = new_pages
    mark_pending(state, changed, deleted)
    save_state(settings.sync_state_file, state)

    console.print(f"\n[bold green]Listo:[/bold green] {len(new_pages)} páginas, "
                  f"{len(changed)} nuevas/actualizadas, {len(deleted)} eliminadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
