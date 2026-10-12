"""Formato de fuentes: qué fragmentos citó el modelo, con enlace a Notion."""
import re

from src.retriever import Hit


def cited_numbers(answer: str, total: int) -> list[int]:
    found: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for n in (int(x) for x in re.split(r"\s*,\s*", group)):
            if 1 <= n <= total and n not in found:
                found.append(n)
    return found


def format_sources(hits: list[Hit], answer: str) -> tuple[str, list[str]]:
    """Devuelve (encabezado, líneas). Si el modelo no citó nada, lista lo consultado."""
    nums = cited_numbers(answer, len(hits))
    header = "Fuentes citadas:" if nums else "Fragmentos consultados:"
    chosen = nums or list(range(1, len(hits) + 1))
    lines = []
    for n in chosen:
        h = hits[n - 1]
        where = f"{h.path} > {h.section}" if h.section else h.path
        lines.append(f"[{n}] {where} — {h.url}")
    return header, lines
