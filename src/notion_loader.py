"""Cliente de Notion y recorrido recursivo de una página raíz."""
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import requests

from src.images import ImageRef
from src.notion_markdown import MarkdownRenderer

API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


class NotionError(Exception):
    pass


@dataclass
class NotionPage:
    id: str                 # sin guiones
    title: str
    path: list[str]         # ancestros + título
    url: str
    last_edited: str
    markdown: str = ""
    images: list = field(default_factory=list)   # list[ImageRef]

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


@dataclass
class _Meta:
    id: str
    title: str
    url: str
    last_edited: str
    ancestors: list[str] = field(default_factory=list)


def clean_id(page_id: str) -> str:
    return page_id.replace("-", "").strip()


class NotionClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> dict:
        for attempt in range(6):
            try:
                r = self.session.request(method, f"{API}{path}", timeout=30, **kwargs)
            except requests.RequestException as e:
                raise NotionError(f"No se pudo conectar con Notion: {e.__class__.__name__}. "
                                  "Revisa tu conexión a internet.")
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)))
                continue
            if r.status_code >= 500:
                time.sleep(0.5 * 2 ** attempt)
                continue
            if r.status_code == 401:
                raise NotionError("Token inválido (401). Revisa NOTION_TOKEN.")
            if r.status_code in (403, 404):
                raise NotionError(
                    f"Sin acceso a {path} ({r.status_code}). ¿Compartiste la página "
                    "raíz con tu integración? (··· > Connections)")
            if r.status_code >= 400:
                raise NotionError(f"Error {r.status_code} en {path}: {r.text[:200]}")
            time.sleep(0.12)  # ~3 peticiones/s: respeta el límite de Notion
            return r.json()
        raise NotionError(f"Demasiados reintentos en {path}")

    def get_page(self, page_id: str) -> dict:
        return self._request("GET", f"/pages/{page_id}")

    def children(self, block_id: str) -> list:
        results, cursor = [], None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = self._request("GET", f"/blocks/{block_id}/children", params=params)
            results += data["results"]
            if not data.get("has_more"):
                return results
            cursor = data["next_cursor"]

    def query_database(self, database_id: str) -> list:
        results, cursor = [], None
        while True:
            body = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{database_id}/query", json=body)
            results += data["results"]
            if not data.get("has_more"):
                return results
            cursor = data["next_cursor"]


def _plain(rich_text: list | None) -> str:
    """Texto plano y seguro para usar como descripción dentro de ![...](...)."""
    text = "".join(t.get("plain_text", "") for t in rich_text or [])
    text = re.sub(r"[\[\]()\n\r]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_of(page_obj: dict) -> str:
    for prop in page_obj.get("properties", {}).values():
        if prop.get("type") == "title":
            text = "".join(t.get("plain_text", "") for t in prop.get("title", []))
            return text.strip() or "Sin título"
    return "Sin título"


def crawl(client: NotionClient, root_id: str) -> Iterator[NotionPage]:
    """Recorre la página raíz y todas sus subpáginas (y bases de datos hijas)."""
    root = client.get_page(root_id)
    queue = deque([_Meta(clean_id(root["id"]), _title_of(root),
                         root["url"], root["last_edited_time"])])
    visited: set[str] = set()

    while queue:
        meta = queue.popleft()
        if meta.id in visited:
            continue
        visited.add(meta.id)

        found: list[tuple[str, dict]] = []
        images: list[ImageRef] = []

        def on_image(block: dict) -> str:
            data = block.get("image", {})
            url = (data.get("file") or data.get("external") or {}).get("url", "")
            if not url:
                return ""
            ref = ImageRef(clean_id(block["id"]), url, _plain(data.get("caption")))
            images.append(ref)
            return f"![{ref.caption}](img:{ref.id})"

        renderer = MarkdownRenderer(
            client.children,
            on_child_page=lambda b: found.append(("page", b)),
            on_child_database=lambda b: found.append(("db", b)),
            on_image=on_image,
        )
        markdown = renderer.render(client.children(meta.id))
        path = meta.ancestors + [meta.title]
        yield NotionPage(meta.id, meta.title, path, meta.url,
                         meta.last_edited, markdown, images)

        for kind, block in found:
            if kind == "page":
                bid = clean_id(block["id"])
                queue.append(_Meta(bid, block["child_page"]["title"] or "Sin título",
                                   f"https://www.notion.so/{bid}",
                                   block["last_edited_time"], path))
            else:
                db_title = block["child_database"]["title"] or "Base de datos"
                for row in client.query_database(block["id"]):
                    queue.append(_Meta(clean_id(row["id"]), _title_of(row),
                                       row["url"], row["last_edited_time"],
                                       path + [db_title]))
