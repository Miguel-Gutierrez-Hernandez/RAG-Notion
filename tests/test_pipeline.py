"""Pruebas offline de sync+indexado y del filtro por varias páginas.  python3 -m tests.test_pipeline"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.hybrid_retriever import HybridRetriever
from src.indexer import Indexer
from src.pipeline import run_index, run_sync
from src.retriever import Retriever
from tests.test_index import FakeEmbedder
from tests.test_qa import build_index
from tests.test_sync import FakeClient


def make_settings(tmp):
    return SimpleNamespace(notion_token="x", notion_root_page_id="root",
                           raw_dir=tmp / "raw", sync_state_file=tmp / "state.json",
                           validate_notion=lambda: [])


def test_sync_and_index():
    tmp = Path(tempfile.mkdtemp())
    s = make_settings(tmp)
    msgs = []
    r = run_sync(s, client=FakeClient(), progress=msgs.append)
    assert (r.total, r.changed, r.deleted) == (4, 4, 0) and len(msgs) == 4

    idx = Indexer(tmp / "vs", FakeEmbedder(), 300, 50)
    assert run_index(s, idx).indexed == 4 and idx.count() >= 4
    assert run_index(s, idx).indexed == 0            # nada pendiente

    r2 = run_sync(s, client=FakeClient(edited="t2", with_child_b=False))
    assert (r2.changed, r2.deleted) == (1, 1)
    ir = run_index(s, idx)
    assert (ir.indexed, ir.deleted) == (1, 1)
    assert all(h.page_id != "b" for h in HybridRetriever(idx).search("contenido de b", k=10))

    assert run_index(s, idx, rebuild=True).indexed == 3
    assert json.loads((tmp / "state.json").read_text())["synced_at"]


def test_sync_failure_keeps_everything():
    tmp = Path(tempfile.mkdtemp())
    s = make_settings(tmp)
    run_sync(s, client=FakeClient())

    class Broken(FakeClient):
        def children(self, bid):
            if bid == "a":
                raise RuntimeError("fallo de red")
            return super().children(bid)

    try:
        run_sync(s, client=Broken(edited="t9"))
        raise AssertionError("debía fallar")
    except RuntimeError:
        pass
    st = json.loads((tmp / "state.json").read_text())
    assert len(st["pages"]) == 4 and (tmp / "raw" / "b.md").exists()


def test_filter_by_list():
    idx = build_index()
    hits = HybridRetriever(idx, top_k=3).search("gatos", k=3, page_id=["p2", "p3"])
    assert {h.page_id for h in hits} <= {"p2", "p3"}
    assert Retriever(idx, 3).search("gatos", page_id=["p1"])[0].page_id == "p1"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("OK")
