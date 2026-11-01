"""Divide una página Markdown en fragmentos (chunks) con contexto."""
import json
import re
from dataclasses import dataclass

FENCE = "```"
# Marcadores de imagen/archivo sin descripción: no aportan texto y ensucian la búsqueda.
PLACEHOLDER_RE = re.compile(r"(?m)^[ \t]*\[(?:Imagen|Archivo|PDF|Video|Audio)\][ \t]*\n?")
# Imágenes: ![descripción](img:ID) -> marcador sin espacios que no se parte al trocear
IMG_MD = re.compile(r"!\[([^\]]*)\]\(img:([0-9a-fA-F-]+)\)")
IMG_TOKEN = re.compile(r"⟦IMG:([0-9a-f]+)⟧")


@dataclass
class Chunk:
    id: str
    page_id: str
    content: str       # texto del fragmento (lo que se muestra y se pasa al LLM)
    embed_text: str    # contenido + ruta/sección (lo que se vectoriza)
    metadata: dict


def parse_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            meta = {}
            for line in text[4:end].splitlines():
                if ": " in line:
                    key, value = line.split(": ", 1)
                    meta[key.strip()] = value.strip()
            return meta, text[end + 5:].lstrip("\n")
    return {}, text


def split_sections(body: str, title: str = "") -> list[tuple[str, str]]:
    """Devuelve [(ruta_de_encabezados, texto)] respetando bloques de código."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False
    started = False

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append((" > ".join(h for _, h in stack), text))
        buf.clear()

    for line in body.splitlines():
        if line.lstrip().startswith(FENCE):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if not started and level == 1 and heading == title:
                started = True
                continue  # el título de la página ya va en la ruta
            started = True
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            if line.strip():
                started = True
            buf.append(line)
    flush()
    return sections


def split_blocks(text: str) -> list[str]:
    """Separa por líneas en blanco sin partir bloques de código."""
    blocks, cur, in_code = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith(FENCE):
            in_code = not in_code
        if not line.strip() and not in_code:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _hard_split(text: str, size: int) -> list[str]:
    parts = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def split_long(block: str, size: int) -> list[str]:
    """Parte un bloque más grande que `size` por líneas, frases y, si hace falta, palabras."""
    if len(block) <= size:
        return [block]
    units: list[str] = []
    for line in block.split("\n"):
        if len(line) <= size:
            units.append(line)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            units.extend(_hard_split(sentence, size) if len(sentence) > size else [sentence])
    pieces, cur = [], ""
    for u in units:
        if cur and len(cur) + 1 + len(u) > size:
            pieces.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        pieces.append(cur)
    return pieces


def _tail(text: str, n: int) -> str:
    if n <= 0 or len(text) <= n:
        return ""
    tail = text[-n:]
    m = re.search(r"\s", tail)
    return tail[m.end():] if m else tail


def pack(blocks: list[str], size: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for block in blocks:
        for piece in split_long(block, size):
            if cur and len(cur) + 2 + len(piece) > size:
                chunks.append(cur)
                tail = _tail(cur, overlap)
                cur = f"{tail}\n\n{piece}" if tail else piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def protect_images(body: str) -> tuple[str, dict]:
    captions: dict[str, str] = {}

    def repl(m):
        image_id = m.group(2).replace("-", "").lower()
        captions.setdefault(image_id, m.group(1).strip())
        return f"⟦IMG:{image_id}⟧"

    return IMG_MD.sub(repl, body), captions


def restore_images(text: str, captions: dict) -> tuple[str, list[dict]]:
    """Cambia los marcadores por texto legible y devuelve las imágenes del fragmento."""
    images: list[dict] = []

    def repl(m):
        image_id = m.group(1).lower()
        caption = captions.get(image_id, "")
        if image_id not in [im["id"] for im in images]:
            images.append({"id": image_id, "caption": caption})
        return f"[Imagen: {caption}]" if caption else "[Imagen]"

    return IMG_TOKEN.sub(repl, text), images


def chunk_page(md_text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    meta, body = parse_front_matter(md_text)
    body = PLACEHOLDER_RE.sub("", body)
    body, captions = protect_images(body)
    page_id = meta.get("id", "")
    title = meta.get("title", "")
    path = meta.get("path", title)

    chunks: list[Chunk] = []
    for section, text in split_sections(body, title):
        for content in pack(split_blocks(text), size, overlap):
            content, images = restore_images(content, captions)
            context = f"{path} > {section}" if section else path
            i = len(chunks)
            chunks.append(Chunk(
                id=f"{page_id}:{i}",
                page_id=page_id,
                content=content,
                embed_text=f"{context}\n\n{content}",
                metadata={"page_id": page_id, "title": title, "path": path,
                          "section": section, "url": meta.get("url", ""),
                          "last_edited": meta.get("last_edited", ""),
                          "chunk_index": i,
                          "images": json.dumps(images, ensure_ascii=False)},
            ))
    return chunks
