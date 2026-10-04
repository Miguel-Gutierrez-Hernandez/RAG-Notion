"""Prueba offline del recorrido y la sincronización incremental.  python3 -m tests.test_sync"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import scripts.sync_notion as sn
from src.notion_loader import crawl


def rt(t):
    return [{"type": "text", "plain_text": t, "annotations": {}, "href": None}]


class FakeClient:
    def __init__(self, edited="t1", with_child_b=True):
        self.edited, self.with_b = edited, with_child_b

    def get_page(self, pid):
        return {"id": "root", "url": "https://notion.so/root", "last_edited_time": "t1",
                "properties": {"title": {"type": "title", "title": rt("Raíz")}}}

    def children(self, bid):
        if bid == "root":
            kids = [{"id": "a", "type": "child_page", "has_children": True,
                     "last_edited_time": self.edited, "child_page": {"title": "Hija A"}},
                    {"id": "p", "type": "paragraph", "has_children": False,
                     "paragraph": {"rich_text": rt("texto raíz")}}]
            if self.with_b:
                kids.append({"id": "b", "type": "child_page", "has_children": True,
                             "last_edited_time": "t1", "child_page": {"title": "Hija B"}})
            return kids
        if bid == "a":
            return [{"id": "aa", "type": "child_page", "has_children": True,
                     "last_edited_time": "t1", "child_page": {"title": "Nieta"}}]
        return [{"id": "x" + bid, "type": "paragraph", "has_children": False,
                 "paragraph": {"rich_text": rt("contenido de " + bid)}}]


def run(client, tmp):
    sn.settings = SimpleNamespace(
        notion_token="x", notion_root_page_id="root", validate_notion=lambda: [],
        raw_dir=tmp / "raw", sync_state_file=tmp / "state.json")
    sn.NotionClient = lambda token: client
    sn.crawl = lambda c, r: crawl(c, r)
    return sn.main.__wrapped__() if hasattr(sn.main, "__wrapped__") else sn.main()


def test_all():
    import sys
    sys.argv = ["sync"]
    tmp = Path(tempfile.mkdtemp())
    pages = list(crawl(FakeClient(), "root"))
    assert [p.path_str for p in pages] == ["Raíz", "Raíz > Hija A", "Raíz > Hija B", "Raíz > Hija A > Nieta"]
    assert "texto raíz" in pages[0].markdown

    assert run(FakeClient(), tmp) == 0
    import json
    st = json.loads((tmp / "state.json").read_text())
    assert len(st["pages"]) == 4 and len(st["pending"]["changed"]) == 4
    assert len(list((tmp / "raw").glob("*.md"))) == 4

    # Segunda sync: Hija A editada, Hija B borrada
    assert run(FakeClient(edited="t2", with_child_b=False), tmp) == 0
    st = json.loads((tmp / "state.json").read_text())
    assert st["pending"]["deleted"] == ["b"], st["pending"]
    assert "a" in st["pending"]["changed"]
    assert not (tmp / "raw" / "b.md").exists()
    print("OK")


if __name__ == "__main__":
    test_all()
