"""Prueba offline del recorrido de páginas.  python3 -m tests.test_sync"""
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


def test_crawl():
    pages = list(crawl(FakeClient(), "root"))
    assert [p.path_str for p in pages] == ["Raíz", "Raíz > Hija A", "Raíz > Hija B", "Raíz > Hija A > Nieta"]
    assert "texto raíz" in pages[0].markdown


if __name__ == "__main__":
    test_crawl()
    print("OK")
