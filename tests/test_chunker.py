"""Prueba offline del chunker.  python3 -m tests.test_chunker"""
from src.chunker import chunk_page

DOC = """---
id: abc123
title: Mascotas
path: Raíz > Mascotas
url: https://notion.so/abc123
last_edited: 2026-01-01T00:00:00Z
---

# Mascotas

Intro corta.

## Gatos

Los gatos comen dos veces al día.

```python
def alimentar():

    return "pienso"
```

## Perros

""" + ("Los perros necesitan paseo diario y mucha atención. " * 40) + "\n"


def test_chunks():
    chunks = chunk_page(DOC, size=300, overlap=50)
    sections = {c.metadata["section"] for c in chunks}
    assert "Gatos" in sections and "Perros" in sections
    assert not any(c.content.startswith("# Mascotas") for c in chunks)  # título omitido
    code = [c for c in chunks if "def alimentar" in c.content]
    assert len(code) == 1 and 'return "pienso"' in code[0].content  # código intacto
    perros = [c for c in chunks if c.metadata["section"] == "Perros"]
    assert len(perros) > 1 and all(len(c.content) <= 300 + 50 + 2 for c in perros)
    assert all(c.embed_text.startswith("Raíz > Mascotas") for c in chunks)
    assert all(c.metadata["page_id"] == "abc123" and c.metadata["url"] for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    print(f"OK ({len(chunks)} fragmentos)")


if __name__ == "__main__":
    test_chunks()
