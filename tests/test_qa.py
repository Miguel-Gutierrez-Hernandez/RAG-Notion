"""Pruebas offline: BM25, búsqueda híbrida, cadena QA, citas y cliente LLM.  python3 -m tests.test_qa"""
import json
import tempfile
from pathlib import Path

import requests

from src.citations import cited_numbers, format_sources
from src.hybrid_retriever import BM25, HybridRetriever, tokenize
from src.indexer import Indexer
from src.llm import LLMError, OllamaChat
from src.qa_chain import QAChain
from tests.test_index import FakeEmbedder, page


def test_tokenize():
    assert tokenize("Los gatos COMEN pescado") == ["gato", "comen", "pescado"]
    assert tokenize("Cañón ciudades") == ["canon", "ciudad"]


def test_bm25_exact_term():
    docs = [tokenize("política de vacaciones general"), tokenize("código ZX-4471 de acceso")]
    scores = BM25(docs).scores(tokenize("zx 4471"))
    assert max(scores, key=scores.get) == 1


def build_index():
    idx = Indexer(Path(tempfile.mkdtemp()), FakeEmbedder(), 300, 50)
    idx.index_page(page("p1", "Mascotas", "## Gatos\n\nLos gatos comen pienso dos veces al día."))
    idx.index_page(page("p2", "Oficina", "## Acceso\n\nEl código de la puerta es ZX4471 y cambia cada mes."))
    idx.index_page(page("p3", "Viajes", "## Japón\n\nEl viaje a Japón será en primavera con vuelo directo."))
    return idx


def test_hybrid():
    idx = build_index()
    h = HybridRetriever(idx, top_k=2)
    assert h.search("qué comen los gatos")[0].page_id == "p1"
    assert h.search("ZX4471")[0].page_id == "p2"
    assert h.search("puerta", page_id="p2")[0].page_id == "p2"
    # reindexar refresca el índice de palabras
    idx.index_page(page("p3", "Viajes", "## Japón\n\nAhora el destino es Islandia en invierno.").replace("last_edited: t", "last_edited: t2"))
    assert h.search("Islandia")[0].page_id == "p3"
    assert HybridRetriever(Indexer(Path(tempfile.mkdtemp()), FakeEmbedder())).search("x") == []


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    def stream(self, messages, **options):
        self.calls.append((messages, options))
        if options.get("num_predict") == 80:   # reescritura
            yield "¿Qué comen los gatos "
            yield "dos veces al día?\nextra"
        else:
            yield "Comen pienso "
            yield "[1]."


def test_chain():
    idx = build_index()
    llm = FakeLLM()
    chain = QAChain(HybridRetriever(idx, top_k=2), llm, history_turns=1)

    ans = chain.ask("¿qué comen los gatos?")
    text = "".join(ans.tokens)
    messages, _ = llm.calls[0]
    assert messages[0]["role"] == "system" and "[1]" in messages[-1]["content"]
    assert "¿qué comen los gatos?" in messages[-1]["content"]
    assert len(llm.calls) == 1                      # sin historial: no se reescribe

    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
               {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    ans2 = chain.ask("¿y cuántas veces?", history)
    "".join(ans2.tokens)
    assert ans2.question == "¿Qué comen los gatos dos veces al día?"   # primera línea
    final_messages, _ = llm.calls[-1]
    assert [m["content"] for m in final_messages[1:3]] == ["c", "d"]   # solo 1 turno de historial

    empty = QAChain(HybridRetriever(Indexer(Path(tempfile.mkdtemp()), FakeEmbedder())), llm)
    assert "índice está vacío" in "".join(empty.ask("hola").tokens)

    header, lines = format_sources(ans.hits, text)
    assert header == "Fuentes citadas:" and lines[0].startswith("[1] ")


def test_citations():
    assert cited_numbers("A [1][3] y [2, 3] pero [9].", 3) == [1, 3, 2]
    from src.retriever import Hit
    hits = [Hit("c", "t", "R > t", "S", "u", 0.1, "p")]
    assert format_sources(hits, "sin citas")[0] == "Fragmentos consultados:"


class FakeResp:
    def __init__(self, status, lines):
        self.status_code, self._lines, self.text = status, lines, "err"

    def iter_lines(self):
        return iter(self._lines)


def test_llm_stream(monkeypatch=None):
    orig = requests.post
    try:
        lines = [json.dumps({"message": {"content": "Ho"}}).encode(),
                 json.dumps({"message": {"content": "la"}, "done": False}).encode(),
                 b"", json.dumps({"message": {"content": ""}, "done": True}).encode()]
        requests.post = lambda *a, **k: FakeResp(200, lines)
        assert "".join(OllamaChat("http://x", "m").stream([])) == "Hola"

        requests.post = lambda *a, **k: FakeResp(404, [])
        try:
            list(OllamaChat("http://x", "m").stream([]))
            raise AssertionError("debía fallar")
        except LLMError as e:
            assert "ollama pull m" in str(e)

        def boom(*a, **k):
            raise requests.ConnectionError()
        requests.post = boom
        try:
            list(OllamaChat("http://x", "m").stream([]))
            raise AssertionError("debía fallar")
        except LLMError as e:
            assert "no responde" in str(e)
    finally:
        requests.post = orig


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("OK")
