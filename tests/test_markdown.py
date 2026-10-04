"""Prueba offline del conversor (sin conexión a Notion).  python3 -m tests.test_markdown"""
from src.notion_markdown import MarkdownRenderer


def rt(text, **ann):
    return [{"type": "text", "plain_text": text, "href": ann.pop("href", None),
             "annotations": ann}]


def blk(i, t, data=None, has_children=False):
    return {"id": i, "type": t, "has_children": has_children, t: data or {}}


CHILDREN = {
    "toggle1": [blk("c1", "paragraph", {"rich_text": rt("dentro del toggle")})],
    "tbl": [{"id": "r1", "type": "table_row", "table_row": {"cells": [rt("A"), rt("B")]}},
            {"id": "r2", "type": "table_row", "table_row": {"cells": [rt("1"), rt("2")]}}],
}


def test_render():
    pages = []
    r = MarkdownRenderer(lambda i: CHILDREN[i], on_child_page=pages.append)
    blocks = [
        blk("1", "heading_1", {"rich_text": rt("Título")}),
        blk("2", "paragraph", {"rich_text": rt("Hola ") + rt("mundo", bold=True) + rt(" y ") + rt("web", href="https://x.com")}),
        blk("3", "bulleted_list_item", {"rich_text": rt("uno")}),
        blk("4", "bulleted_list_item", {"rich_text": rt("dos")}),
        blk("5", "numbered_list_item", {"rich_text": rt("primero")}),
        blk("6", "numbered_list_item", {"rich_text": rt("segundo")}),
        blk("7", "to_do", {"rich_text": rt("tarea"), "checked": True}),
        blk("toggle1", "toggle", {"rich_text": rt("más")}, has_children=True),
        blk("9", "code", {"rich_text": rt("print(1)"), "language": "python"}),
        blk("tbl", "table", {}, has_children=True),
        blk("p1", "child_page", {"title": "Subpágina uno"}),
    ]
    md = r.render(blocks)
    print(md)
    assert "# Título" in md
    assert "Hola **mundo** y [web](https://x.com)" in md
    assert "- uno\n- dos" in md
    assert "1. primero\n2. segundo" in md
    assert "- [x] tarea" in md
    assert "- más\n  dentro del toggle" in md
    assert "```python\nprint(1)\n```" in md
    assert "| A | B |" in md and "| 1 | 2 |" in md
    assert len(pages) == 1 and pages[0]["id"] == "p1"
    print("\nOK")


if __name__ == "__main__":
    test_render()
