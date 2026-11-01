"""Pruebas offline de imágenes: lectura, descarga, troceado, recuperación y selección.
python3 -m tests.test_images"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import requests

import src.images as images_mod
from src.chunker import chunk_page
from src.hybrid_retriever import HybridRetriever
from src.images import (ImageRef, cleanup_images, collect_answer_images, download_image,
                        find_image, wants_images)
from src.indexer import Indexer
from src.notion_loader import crawl
from src.notion_markdown import MarkdownRenderer
from src.pipeline import run_index, run_sync
from src.retriever import Hit
from tests.test_index import FakeEmbedder
from tests.test_sync import FakeClient, rt


def image_block(bid="face01", url="https://x/a.png", caption="Mapa [norte]"):
    return {"id": bid, "type": "image", "has_children": False,
            "image": {"type": "file", "file": {"url": url}, "caption": rt(caption)}}


class ImgClient(FakeClient):
    def __init__(self, with_image=True, **kw):
        super().__init__(**kw)
        self.with_image = with_image

    def children(self, bid):
        if bid == "a":
            kids = super().children(bid)
            return ([image_block()] if self.with_image else []) + kids
        return super().children(bid)


class FakeResp:
    def __init__(self, status=200, ctype="image/png", data=b"PNGDATA"):
        self.status_code, self.headers, self.data = status, {"Content-Type": ctype}, data

    def iter_content(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i:i + n]

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, **kw):
        r = self.responses[url]
        if isinstance(r, Exception):
            raise r
        return r


def test_renderer():
    r = MarkdownRenderer(lambda i: [], on_image=lambda b: f"![x](img:{b['id']})")
    assert r.render([image_block("ab12")]) == "![x](img:ab12)"
    assert MarkdownRenderer(lambda i: []).render([image_block()]).startswith("[Imagen")


def test_crawl_collects_images():
    pages = list(crawl(ImgClient(), "root"))
    a = [p for p in pages if p.id == "a"][0]
    assert a.images == [ImageRef("face01", "https://x/a.png", "Mapa norte")]
    assert "![Mapa norte](img:face01)" in a.markdown
    assert all(not p.images for p in pages if p.id != "a")


DOC = """---
id: pg1
title: Ciudades
path: Ciudades
url: https://n.so/pg1
last_edited: t
---

# Ciudades

## Puerto Gris

Ciudad portuaria.

![Mapa de Puerto Gris](img:aaaa1111)

## Galería

![](img:bbbb2222)

## Vacía

[Imagen]
"""


def test_chunker_images():
    chunks = chunk_page(DOC)
    secs = {c.metadata["section"]: c for c in chunks}
    c = secs["Puerto Gris"]
    assert json.loads(c.metadata["images"]) == [{"id": "aaaa1111", "caption": "Mapa de Puerto Gris"}]
    assert "[Imagen: Mapa de Puerto Gris]" in c.content and "⟦" not in c.embed_text
    g = secs["Galería"]                       # sección con solo una imagen: sigue siendo buscable
    assert g.content == "[Imagen]" and json.loads(g.metadata["images"]) == [{"id": "bbbb2222", "caption": ""}]
    assert "Vacía" not in secs                # marcador heredado sin imagen real

    long_doc = DOC.split("## Puerto")[0] + "## Largo\n\n" + "Texto largo. " * 120 \
        + "\n\n![Mapa](img:cccc3333)\n\n" + "Más texto. " * 120 + "\n"
    chunks = chunk_page(long_doc, size=300, overlap=60)
    assert len(chunks) > 3
    assert all("⟦" not in c.content and "(img:" not in c.content and "IMG:" not in c.content for c in chunks)
    assert any("cccc3333" in c.metadata["images"] for c in chunks)


def test_download():
    tmp = Path(tempfile.mkdtemp())
    ref = ImageRef("aa11", "https://x/a.png", "m")
    sess = FakeSession({"https://x/a.png": FakeResp(data=b"PNGDATA")})
    assert download_image(ref, tmp, sess) == "ok"
    assert (tmp / "aa11.png").read_bytes() == b"PNGDATA" and find_image(tmp, "aa11")
    assert download_image(ref, tmp, sess) == "existente"

    sess.responses["https://x/a.png"] = FakeResp(ctype="image/jpeg", data=b"JPG")
    assert download_image(ref, tmp, sess, overwrite=True) == "ok"
    assert not (tmp / "aa11.png").exists() and (tmp / "aa11.jpg").read_bytes() == b"JPG"

    other = FakeSession({"https://x/c.webp": FakeResp(ctype="application/octet-stream"),
                         "https://x/x.bin": FakeResp(ctype="application/octet-stream"),
                         "https://x/no.png": FakeResp(status=403),
                         "https://x/red.png": requests.ConnectionError()})
    assert download_image(ImageRef("cc33", "https://x/c.webp"), tmp, other) == "ok"
    assert (tmp / "cc33.webp").exists()
    assert download_image(ImageRef("dd44", "https://x/x.bin"), tmp, other) == "formato no soportado"
    assert download_image(ImageRef("ee55", "https://x/no.png"), tmp, other) == "HTTP 403"
    assert "ConnectionError" in download_image(ImageRef("ff66", "https://x/red.png"), tmp, other)

    old = images_mod.MAX_BYTES
    images_mod.MAX_BYTES = 5
    try:
        assert download_image(ImageRef("ab99", "https://x/a.png"), tmp,
                              FakeSession({"https://x/a.png": FakeResp(data=b"x" * 50)})) == "demasiado grande"
    finally:
        images_mod.MAX_BYTES = old
    assert not list(tmp.glob("*.part")) and not (tmp / "ab99.png").exists()

    (tmp / "zz99.png").write_bytes(b"x")
    removed = cleanup_images(tmp, {"aa11", "cc33"})
    assert removed >= 1 and not (tmp / "zz99.png").exists() and (tmp / "aa11.jpg").exists()


def make_hits():
    return [
        Hit("c", "t", "P", "Sec1", "u1", 0.1, "p1", id="1", images=[{"id": "i1", "caption": "Uno"}]),
        Hit("c", "t", "P", "Sec2", "u2", 0.1, "p2", id="2",
            images=[{"id": "i2", "caption": "Dos"}, {"id": "i1", "caption": "Uno"}]),
        Hit("c", "t", "P", "Sec3", "u3", 0.1, "p3", id="3", images=[{"id": "i3", "caption": "Tres"}]),
    ]


def test_collect_answer_images():
    d = Path(tempfile.mkdtemp())
    for name in ("i1.png", "i2.png"):
        (d / name).write_bytes(b"x")
    hits = make_hits()
    ids = lambda r: [x["id"] for x in r]

    r = collect_answer_images(hits, "Respuesta [2].", "¿qué hay?", d)
    assert ids(r) == ["i2", "i1"] and r[0]["label"] == "P > Sec2" and r[0]["url"] == "u2"
    assert ids(collect_answer_images(hits, "Sin citas.", "¿qué hay?", d)) == ["i1", "i2"]   # 2 primeros
    r = collect_answer_images(hits, "Mira [3].", "Muéstrame el mapa", d)                    # lo pides
    assert ids(r) == ["i3"] and r[0]["path"] is None          # solo lo citado; no se rellena con otras
    assert ids(collect_answer_images(hits, "Sin citas.", "Muéstrame el mapa", d)) == ["i1", "i2", "i3"]
    no_img = [Hit("c", "t", "P", "S", "u", 0.1, "p", id="z")] + hits
    assert ids(collect_answer_images(no_img, "Mira [1].", "Dame una imagen", d)) == ["i1", "i2"]   # los 3 primeros
    assert collect_answer_images(no_img, "Mira [1].", "¿Qué hay?", d) == []                 # sin pedirlo, no se rellena
    assert collect_answer_images(hits, "No encuentro esa información en tus notas.", "¿qué hay?", d) == []
    assert collect_answer_images(hits, "[1][2][3]", "dame imágenes", d, limit=2).__len__() == 2
    assert wants_images("Enséñame una foto del dragón") and wants_images("¿Hay un mapa?")
    assert not wants_images("¿Cuánta vida tiene el dragón?")


def page_md(pid, body):
    return (f"---\nid: {pid}\ntitle: Ciudades\npath: Ciudades\nurl: https://n.so/{pid}\n"
            f"last_edited: t\n---\n\n# Ciudades\n\n{body}\n")


def test_retrieval_returns_images():
    idx = Indexer(Path(tempfile.mkdtemp()), FakeEmbedder(), 300, 50)
    idx.index_page(page_md("pg1", "## Puerto Gris\n\nCiudad portuaria.\n\n![Mapa de Puerto Gris](img:aaaa1111)"))
    idx.index_page(page_md("pg2", "## Bosque\n\nArboles y rios."))
    h = HybridRetriever(idx, top_k=2)
    hit = h.search("mapa de Puerto Gris")[0]
    assert hit.page_id == "pg1" and hit.images == [{"id": "aaaa1111", "caption": "Mapa de Puerto Gris"}]
    assert h._keyword_hits("mapa", None, 5)[0].images[0]["id"] == "aaaa1111"   # camino por palabras
    assert h.search("bosque")[0].images == []


def test_sync_downloads_and_cleans():
    tmp = Path(tempfile.mkdtemp())
    s = SimpleNamespace(notion_token="x", notion_root_page_id="root", raw_dir=tmp / "raw",
                        images_dir=tmp / "images", sync_state_file=tmp / "state.json",
                        validate_notion=lambda: [])
    sess = FakeSession({"https://x/a.png": FakeResp()})
    r = run_sync(s, client=ImgClient(), image_session=sess)
    assert r.images == 1 and (tmp / "images" / "face01.png").exists()
    assert "![Mapa norte](img:face01)" in (tmp / "raw" / "a.md").read_text(encoding="utf-8")

    r2 = run_sync(s, client=ImgClient(), image_session=sess)
    assert (r2.changed, r2.images) == (0, 0)

    # archivo descargado antes de existir las imágenes: se reescribe solo
    (tmp / "raw" / "a.md").write_text("---\nid: a\n---\n\nviejo", encoding="utf-8")
    assert run_sync(s, client=ImgClient(), image_session=sess).changed == 1
    assert "img:face01" in (tmp / "raw" / "a.md").read_text(encoding="utf-8")

    # indexado y búsqueda
    idx = Indexer(tmp / "vs", FakeEmbedder(), 300, 50)
    run_index(s, idx)
    assert any(h.images for h in HybridRetriever(idx, top_k=5).search("Mapa norte"))

    # la imagen desaparece de Notion: se borra del disco
    run_sync(s, client=ImgClient(with_image=False), image_session=sess)
    assert not (tmp / "images" / "face01.png").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("OK")
