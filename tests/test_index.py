"""Prueba offline de indexado y búsqueda (embeddings simulados).  python3 -m tests.test_index"""
import re
import tempfile
import zlib
from pathlib import Path

from src.indexer import IndexerError, Indexer
from src.retriever import Retriever


class FakeEmbedder:
    model = "fake"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"\w+", t.lower()):
                v[zlib.crc32(w.encode()) % 256] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


def page(pid, title, body):
    return f"---\nid: {pid}\ntitle: {title}\npath: Raíz > {title}\nurl: https://n.so/{pid}\nlast_edited: t\n---\n\n# {title}\n\n{body}\n"


def test_all():
    tmp = Path(tempfile.mkdtemp())
    idx = Indexer(tmp, FakeEmbedder(), 300, 50)
    idx.index_page(page("p1", "Mascotas", "## Gatos\n\nLos gatos comen pienso dos veces al día."))
    idx.index_page(page("p2", "Finanzas", "## Impuestos\n\nLa declaración de la renta se presenta en junio."))
    r = Retriever(idx, 2)

    assert r.search("¿qué comen los gatos?")[0].page_id == "p1"
    assert r.search("cuándo es la declaración de la renta")[0].page_id == "p2"
    assert r.search("gatos", page_id="p2")[0].page_id == "p2"  # filtro por página

    n = idx.count()
    idx.index_page(page("p1", "Mascotas", "## Gatos\n\nAhora comen pescado."))  # reindexar sustituye
    assert idx.count() == n
    assert "pescado" in r.search("gatos pescado")[0].content

    idx.delete_page("p2")
    assert all(h.page_id == "p1" for h in r.search("renta", k=5))

    idx.reset()
    assert idx.count() == 0

    idx2 = Indexer(tmp, FakeEmbedder(), 300, 50)
    idx2.index_page(page("p3", "X", "texto"))
    other = FakeEmbedder(); other.model = "otro"
    try:
        Indexer(tmp, other)
        raise AssertionError("debía fallar por cambio de modelo")
    except IndexerError:
        pass
    print("OK")


if __name__ == "__main__":
    test_all()
