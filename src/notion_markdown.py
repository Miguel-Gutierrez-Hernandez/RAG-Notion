"""Conversión de bloques de Notion a Markdown."""
from typing import Callable, Optional

MEDIA_LABELS = {"image": "Imagen", "file": "Archivo", "pdf": "PDF",
                "video": "Video", "audio": "Audio"}
LIST_TYPES = {"bulleted_list_item", "numbered_list_item", "to_do"}
PASSTHROUGH = {"column_list", "column", "synced_block", "template"}


def rich_text_to_md(rich_text: Optional[list]) -> str:
    parts = []
    for rt in rich_text or []:
        if rt.get("type") == "equation":
            parts.append(f"${rt['equation']['expression']}$")
            continue
        text = rt.get("plain_text", "")
        if not text.strip():
            parts.append(text)
            continue
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        core = text.strip()
        ann = rt.get("annotations", {})
        if ann.get("code"):
            core = f"`{core}`"
        if ann.get("bold"):
            core = f"**{core}**"
        if ann.get("italic"):
            core = f"*{core}*"
        if ann.get("strikethrough"):
            core = f"~~{core}~~"
        if rt.get("href"):
            core = f"[{core}]({rt['href']})"
        parts.append(f"{lead}{core}{trail}")
    return "".join(parts)


class MarkdownRenderer:
    def __init__(self, fetch_children: Callable[[str], list],
                 on_child_page: Optional[Callable] = None,
                 on_child_database: Optional[Callable] = None):
        self.fetch_children = fetch_children
        self.on_child_page = on_child_page or (lambda b: None)
        self.on_child_database = on_child_database or (lambda b: None)

    def render(self, blocks: list, indent: int = 0) -> str:
        pieces: list[tuple[bool, str]] = []
        number = 0
        for b in blocks:
            t = b.get("type")
            number = number + 1 if t == "numbered_list_item" else 0
            text = self._block(b, t, indent, number)
            if text:
                pieces.append((t in LIST_TYPES, text))
        out = ""
        for i, (is_list, text) in enumerate(pieces):
            if i:
                out += "\n" if (is_list and pieces[i - 1][0]) else "\n\n"
            out += text
        return out

    def _kids(self, b: dict, indent: int) -> str:
        if not b.get("has_children"):
            return ""
        block_id = b["id"]
        if b.get("type") == "synced_block":
            origin = (b["synced_block"].get("synced_from") or {}).get("block_id")
            block_id = origin or block_id
        return self.render(self.fetch_children(block_id), indent)

    @staticmethod
    def _pad(text: str, pad: str) -> str:
        if not pad:
            return text
        return "\n".join(pad + line if line else line for line in text.split("\n"))

    def _join(self, head: str, kids: str) -> str:
        return f"{head}\n{kids}" if head and kids else (head or kids)

    def _block(self, b: dict, t: str, indent: int, number: int) -> str:
        data = b.get(t, {}) or {}
        text = rich_text_to_md(data.get("rich_text"))
        pad = "  " * indent

        if t == "paragraph":
            return self._join(self._pad(text, pad), self._kids(b, indent))
        if t in ("heading_1", "heading_2", "heading_3"):
            level = int(t[-1])
            return self._join(self._pad(f"{'#' * level} {text}", pad), self._kids(b, indent))
        if t in ("bulleted_list_item", "numbered_list_item", "to_do", "toggle"):
            if t == "numbered_list_item":
                marker = f"{number}. "
            elif t == "to_do":
                marker = "- [x] " if data.get("checked") else "- [ ] "
            else:
                marker = "- "
            return self._join(f"{pad}{marker}{text}", self._kids(b, indent + 1))
        if t == "quote":
            return self._join(self._pad(f"> {text}", pad), self._kids(b, indent))
        if t == "callout":
            icon = (data.get("icon") or {}).get("emoji", "")
            return self._join(self._pad(f"> {icon} {text}".replace(">  ", "> "), pad), self._kids(b, indent))
        if t == "code":
            code = "".join(r.get("plain_text", "") for r in data.get("rich_text", []))
            lang = data.get("language", "")
            lang = "" if lang == "plain text" else lang
            return self._pad(f"```{lang}\n{code}\n```", pad)
        if t == "divider":
            return self._pad("---", pad)
        if t == "equation":
            return f"$${data.get('expression', '')}$$"
        if t == "table":
            return self._pad(self._table(b), pad)
        if t == "child_page":
            self.on_child_page(b)
            return self._pad(f"Subpágina: **{data.get('title', '')}**", pad)
        if t == "child_database":
            self.on_child_database(b)
            return self._pad(f"Base de datos: **{data.get('title', '')}**", pad)
        if t in ("bookmark", "embed", "link_preview"):
            cap = rich_text_to_md(data.get("caption"))
            url = data.get("url", "")
            return self._pad(f"{cap}: {url}" if cap else url, pad)
        if t in MEDIA_LABELS:
            cap = rich_text_to_md(data.get("caption"))
            label = MEDIA_LABELS[t]
            return self._pad(f"[{label}: {cap}]" if cap else f"[{label}]", pad)
        if t in PASSTHROUGH:
            return self._kids(b, indent)
        # Tipos no soportados: se conserva el texto si lo hay
        return self._join(self._pad(text, pad), self._kids(b, indent))

    def _table(self, b: dict) -> str:
        rows = [r for r in self.fetch_children(b["id"]) if r.get("type") == "table_row"]
        matrix = [[rich_text_to_md(c).replace("|", "\\|").replace("\n", " ")
                   for c in r["table_row"]["cells"]] for r in rows]
        if not matrix:
            return ""
        lines = ["| " + " | ".join(matrix[0]) + " |",
                 "| " + " | ".join("---" for _ in matrix[0]) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in matrix[1:]]
        return "\n".join(lines)
