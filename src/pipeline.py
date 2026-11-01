"""Sincronizar con Notion e indexar: operaciones reutilizables (las usa la app web)."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.images import cleanup_images, download_image
from src.notion_loader import NotionClient, crawl
from src.sync_state import load_state, mark_pending, save_state


@dataclass
class SyncResult:
    total: int
    changed: int
    deleted: int
    images: int = 0          # imágenes descargadas en esta sincronización


@dataclass
class IndexResult:
    indexed: int
    deleted: int
    chunks: int


def page_markdown(page) -> str:
    header = (f"---\nid: {page.id}\ntitle: {page.title}\npath: {page.path_str}\n"
              f"url: {page.url}\nlast_edited: {page.last_edited}\n---\n\n")
    return f"{header}# {page.title}\n\n{page.markdown}\n"


def _lacks_image_markers(file: Path) -> bool:
    """Archivos descargados antes de que existieran las imágenes: hay que reescribirlos."""
    try:
        return "](img:" not in file.read_text(encoding="utf-8")
    except OSError:
        return True


def run_sync(settings, full: bool = False, client=None,
             progress: Callable[[str], None] | None = None,
             image_session=None) -> SyncResult:
    """Descarga las páginas de Notion a data/raw (y sus imágenes a data/images).
    Lanza NotionError si algo falla; en ese caso no se guarda ni se borra nada."""
    progress = progress or (lambda m: None)
    client = client or NotionClient(settings.notion_token)
    images_dir = Path(getattr(settings, "images_dir", settings.raw_dir.parent / "images"))
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(settings.sync_state_file)
    old = state["pages"]
    new, changed, written = {}, [], {}
    seen_images: set[str] = set()
    downloaded = 0

    for page in crawl(client, settings.notion_root_page_id):
        prev = old.get(page.id)
        file = settings.raw_dir / f"{page.id}.md"
        modified = (prev is None or prev["last_edited"] != page.last_edited
                    or not file.exists() or full
                    or (bool(page.images) and _lacks_image_markers(file)))
        if modified:
            written[page.id] = (file, page_markdown(page))
            changed.append(page.id)
        new[page.id] = {"title": page.title, "path": page.path_str,
                        "url": page.url, "last_edited": page.last_edited}
        tag = "nueva" if prev is None else "actualizada" if modified else "sin cambios"
        progress(f"{page.path_str} — {tag}")

        # Las URLs de Notion caducan: se descargan ahora, con la URL recién obtenida.
        for ref in page.images:
            seen_images.add(ref.id)
            status = download_image(ref, images_dir, session=image_session,
                                    overwrite=modified)
            if status == "ok":
                downloaded += 1
            elif status != "existente":
                progress(f"   imagen «{ref.caption or ref.id[:8]}»: {status}")

    # El recorrido terminó bien: ahora sí se escribe y se borra.
    for file, text in written.values():
        file.write_text(text, encoding="utf-8")
    deleted = [pid for pid in old if pid not in new]
    for pid in deleted:
        (settings.raw_dir / f"{pid}.md").unlink(missing_ok=True)
        progress(f"{old[pid]['path']} — eliminada")
    cleanup_images(images_dir, seen_images)

    state["pages"] = new
    state["synced_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    mark_pending(state, changed, deleted)
    save_state(settings.sync_state_file, state)
    return SyncResult(len(new), len(changed), len(deleted), downloaded)


def run_index(settings, indexer, rebuild: bool = False,
              progress: Callable[[str], None] | None = None) -> IndexResult:
    """Indexa lo pendiente. Guarda el avance tras cada página (reanudable)."""
    progress = progress or (lambda m: None)
    state = load_state(settings.sync_state_file)
    pending = state["pending"]
    if rebuild:
        indexer.reset()
        pending["changed"], pending["deleted"] = list(state["pages"]), []
        save_state(settings.sync_state_file, state)

    deleted = 0
    for pid in list(pending["deleted"]):
        indexer.delete_page(pid)
        pending["deleted"].remove(pid)
        save_state(settings.sync_state_file, state)
        deleted += 1

    indexed = chunks = 0
    todo = list(pending["changed"])
    for n, pid in enumerate(todo, 1):
        file = settings.raw_dir / f"{pid}.md"
        label = state["pages"].get(pid, {}).get("path", pid)
        if not file.exists():
            pending["changed"].remove(pid)
            continue
        count = indexer.index_page(file.read_text(encoding="utf-8"))
        pending["changed"].remove(pid)
        save_state(settings.sync_state_file, state)
        indexed += 1
        chunks += count
        progress(f"[{n}/{len(todo)}] {label} — {count} fragmentos")
    save_state(settings.sync_state_file, state)
    return IndexResult(indexed, deleted, chunks)
