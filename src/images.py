"""Imágenes de Notion: descarga local, localización y selección para las respuestas.

Las URLs de las imágenes subidas a Notion caducan en ~1 hora, así que se descargan
a data/images/ durante la sincronización y la app las muestra desde ahí.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.citations import cited_numbers

MAX_BYTES = 20 * 1024 * 1024
CONTENT_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
                 "image/gif": "gif", "image/webp": "webp"}
URL_EXTENSIONS = {".png": "png", ".jpg": "jpg", ".jpeg": "jpg", ".gif": "gif", ".webp": "webp"}
IMAGE_WORDS = re.compile(
    r"\b(imagen(?:es)?|fotos?|fotograf[ií]as?|mapas?|ilustraci(?:ón|on|ones)|dibujos?|"
    r"retratos?|planos?|croquis|l[aá]minas?)\b", re.IGNORECASE)
NOT_FOUND = "no encuentro esa información"


@dataclass
class ImageRef:
    id: str          # id del bloque de Notion (sin guiones)
    url: str
    caption: str = ""


def find_image(images_dir: Path, image_id: str) -> Path | None:
    for path in Path(images_dir).glob(f"{image_id}.*"):
        if path.suffix != ".part":
            return path
    return None


def _extension(content_type: str, url: str) -> str | None:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in CONTENT_TYPES:
        return CONTENT_TYPES[ct]
    return URL_EXTENSIONS.get(Path(urlparse(url).path).suffix.lower())


def download_image(ref: ImageRef, images_dir: Path, session=None,
                   overwrite: bool = False) -> str:
    """Descarga una imagen. Devuelve 'ok', 'existente' o el motivo del fallo."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    if find_image(images_dir, ref.id) and not overwrite:
        return "existente"
    try:
        r = (session or requests).get(ref.url, timeout=(10, 60), stream=True,
                                      headers={"User-Agent": "notion-rag/1.0"})
    except requests.RequestException as e:
        return f"error de red ({e.__class__.__name__})"
    tmp = images_dir / f"{ref.id}.part"
    try:
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        ext = _extension(r.headers.get("Content-Type", ""), ref.url)
        if not ext:
            return "formato no soportado"
        size, too_big = 0, False
        with open(tmp, "wb") as f:
            for block in r.iter_content(65536):
                size += len(block)
                if size > MAX_BYTES:
                    too_big = True
                    break
                f.write(block)
        if too_big:
            tmp.unlink(missing_ok=True)
            return "demasiado grande"
        for old in images_dir.glob(f"{ref.id}.*"):      # p. ej. cambió de png a jpg
            if old != tmp:
                old.unlink(missing_ok=True)
        tmp.replace(images_dir / f"{ref.id}.{ext}")
        return "ok"
    except (requests.RequestException, OSError) as e:
        tmp.unlink(missing_ok=True)
        return f"error ({e.__class__.__name__})"
    finally:
        close = getattr(r, "close", None)
        if close:
            close()


def cleanup_images(images_dir: Path, keep_ids: set[str]) -> int:
    """Borra las imágenes que ya no aparecen en ninguna página."""
    removed = 0
    if not Path(images_dir).exists():
        return 0
    for path in Path(images_dir).iterdir():
        if path.is_file() and path.name != ".gitkeep" and (
                path.suffix == ".part" or path.stem not in keep_ids):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def resolve_images(images: list[dict], images_dir: Path) -> list[dict]:
    """[{'id','caption'}] -> [{'id','caption','path'|None}] según lo descargado."""
    out = []
    for im in images:
        path = find_image(images_dir, im["id"])
        out.append({"id": im["id"], "caption": im.get("caption", ""),
                    "path": str(path) if path else None})
    return out


def wants_images(question: str) -> bool:
    return bool(IMAGE_WORDS.search(question or ""))


def _label(hit) -> str:
    return f"{hit.path} > {hit.section}" if hit.section else hit.path


def _gather(hits: list, order: list[int], images_dir: Path, limit: int) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for n in order:
        hit = hits[n - 1]
        for im in resolve_images(getattr(hit, "images", []) or [], images_dir):
            if im["id"] in seen:
                continue
            seen.add(im["id"])
            result.append({**im, "label": _label(hit), "url": hit.url})
            if len(result) >= limit:
                return result
    return result


def collect_answer_images(hits: list, answer: str, question: str, images_dir: Path,
                          limit: int = 6) -> list[dict]:
    """Elige qué imágenes acompañan a la respuesta.

    1. Las de los fragmentos que el modelo citó.
    2. Si no citó ninguno (o no tienen imágenes y pediste verlas): las de los primeros
       fragmentos (2, o 3 si pides "muéstrame el mapa...").
    3. Si el modelo dijo que no encuentra la información: ninguna.
    """
    asked = wants_images(question)
    if NOT_FOUND in answer.lower() and not asked:
        return []
    top = list(range(1, min(3 if asked else 2, len(hits)) + 1))
    cited = cited_numbers(answer, len(hits))
    result = _gather(hits, cited or top, images_dir, limit)
    if not result and asked and cited:
        result = _gather(hits, top, images_dir, limit)
    return result
