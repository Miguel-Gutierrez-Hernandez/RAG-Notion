"""Une recuperación + modelo de chat + prompt: el corazón del asistente."""
from dataclasses import dataclass
from typing import Iterator

from src.llm import OllamaChat
from src.prompts import REWRITE_PROMPT, SYSTEM_PROMPT, USER_TEMPLATE, format_context
from src.retriever import Hit

EMPTY_INDEX_MSG = ("El índice está vacío. Ejecuta primero "
                   "`python3 -m scripts.sync_notion` y `python3 -m scripts.index_notion`.")


@dataclass
class Answer:
    question: str          # pregunta final usada para buscar (ya reescrita si hubo historial)
    hits: list[Hit]
    tokens: Iterator[str]  # respuesta en streaming


class QAChain:
    def __init__(self, retriever, llm: OllamaChat, history_turns: int = 3,
                 rewrite: bool = True):
        self.retriever = retriever
        self.llm = llm
        self.history_turns = history_turns
        self.rewrite = rewrite

    def _recent(self, history: list[dict]) -> list[dict]:
        return history[-self.history_turns * 2:] if history else []

    def standalone_question(self, question: str, history: list[dict]) -> str:
        """Convierte un seguimiento ("¿y cuánto cuesta?") en una pregunta autónoma."""
        recent = self._recent(history)
        if not recent or not self.rewrite:
            return question
        transcript = "\n".join(
            f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:300]}"
            for m in recent)
        prompt = REWRITE_PROMPT.format(history=transcript, question=question)
        out = "".join(self.llm.stream([{"role": "user", "content": prompt}],
                                      temperature=0, num_predict=80)).strip()
        first_line = out.splitlines()[0].strip(' "') if out else ""
        return first_line or question

    def ask(self, question: str, history: list[dict] | None = None,
            page_id: str | list[str] | None = None) -> Answer:
        history = history or []
        search_q = self.standalone_question(question, history)
        hits = self.retriever.search(search_q, page_id=page_id)
        if not hits:
            return Answer(search_q, [], iter([EMPTY_INDEX_MSG]))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self._recent(history)
        messages.append({"role": "user", "content": USER_TEMPLATE.format(
            context=format_context(hits), question=question)})
        return Answer(search_q, hits, self.llm.stream(messages))
