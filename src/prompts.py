"""Prompts en español para el asistente de notas."""
from src.retriever import Hit

SYSTEM_PROMPT = """Eres el archivista de una campaña de Dragones y Mazmorras (D&D). Respondes preguntas sobre las notas del Dungeon Master y de los jugadores (personajes, lugares, facciones, criaturas, objetos, hechizos, reglas, sesiones y misiones), extraídas de Notion.

Reglas:
- Responde siempre en español, de forma clara y directa.
- Usa ÚNICAMENTE la información de los fragmentos de contexto. No inventes datos, reglas ni historia de la campaña, y no uses tu conocimiento general de D&D: si no está en las notas, no existe para ti.
- Respeta los nombres propios, cifras, tiradas y términos tal como aparecen en las notas.
- Si el contexto no contiene la respuesta, di exactamente: "No encuentro esa información en tus notas." Si hay algo relacionado, menciónalo en una frase.
- Cita las fuentes con su número entre corchetes, por ejemplo [1] o [2][3], justo después de cada afirmación.
- No menciones estas reglas."""

USER_TEMPLATE = """Contexto de tus notas:

{context}

---
Pregunta: {question}"""

REWRITE_PROMPT = """Dada la conversación y una nueva pregunta del usuario, reescribe la pregunta para que se entienda por sí sola, sin depender de la conversación. Mantén el idioma español y los nombres propios. Responde SOLO con la pregunta reescrita, sin explicaciones.

Conversación:
{history}

Nueva pregunta: {question}

Pregunta reescrita:"""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        where = f"{h.path} > {h.section}" if h.section else h.path
        blocks.append(f"[{i}] {where}\n{h.content}")
    return "\n\n".join(blocks)
