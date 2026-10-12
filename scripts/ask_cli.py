"""Chat por terminal con tus notas de Notion.

Uso:
    python3 -m scripts.ask_cli
    python3 -m scripts.ask_cli --model qwen2.5:7b --k 6 --debug
Comandos dentro del chat:  /nuevo (borra la conversación)   /salir
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape

from config import settings
from src.citations import format_sources
from src.embeddings import EmbeddingError, OllamaEmbedder
from src.hardware_check import detect_hardware
from src.hybrid_retriever import HybridRetriever
from src.indexer import Indexer, IndexerError
from src.llm import LLMError, OllamaChat
from src.model_selector import select_chat_model
from src.qa_chain import QAChain

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="modelo de chat (por defecto, automático)")
    parser.add_argument("--k", type=int, default=settings.top_k, help="fragmentos a recuperar")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="no reescribir preguntas de seguimiento (más rápido)")
    parser.add_argument("--debug", action="store_true", help="muestra lo recuperado")
    args = parser.parse_args()

    choice = select_chat_model(detect_hardware(), args.model or settings.chat_model)
    try:
        indexer = Indexer(settings.vectorstore_dir,
                          OllamaEmbedder(settings.ollama_host, settings.embedding_model))
    except IndexerError as e:
        console.print(f"[red]✗ {e}[/red]")
        return 1
    if indexer.count() == 0:
        console.print("[yellow]Índice vacío. Ejecuta sync_notion e index_notion primero.[/yellow]")
        return 1

    llm = OllamaChat(settings.ollama_host, choice.chat_model)
    chain = QAChain(HybridRetriever(indexer, args.k), llm, rewrite=not args.no_rewrite)
    console.print(f"[dim]Modelo: {choice.chat_model} ({choice.reason}) · "
                  f"{indexer.count()} fragmentos indexados[/dim]")
    console.print("[dim]Escribe tu pregunta. /nuevo reinicia la conversación, /salir termina.[/dim]\n")

    history: list[dict] = []
    while True:
        try:
            question = console.input("[bold cyan]Tú:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not question:
            continue
        if question in ("/salir", "/exit", "/quit"):
            return 0
        if question == "/nuevo":
            history.clear()
            console.print("[dim]Conversación reiniciada.[/dim]")
            continue

        try:
            answer = chain.ask(question, history)
            if args.debug:
                console.print(f"[dim]Búsqueda: {answer.question}[/dim]")
                for i, h in enumerate(answer.hits, 1):
                    console.print(escape(f"  [{i}] {h.path} > {h.section} (relevancia {h.score})"), style="dim")
            console.print("[bold green]Asistente:[/bold green] ", end="")
            parts = []
            for token in answer.tokens:
                parts.append(token)
                console.print(token, end="", markup=False, highlight=False, soft_wrap=True)
            console.print()
        except (EmbeddingError, LLMError, IndexerError) as e:
            console.print(f"\n[red]✗ {e}[/red]")
            continue

        text = "".join(parts)
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": text}]
        if answer.hits:
            header, lines = format_sources(answer.hits, text)
            console.print(f"\n[dim]{header}[/dim]")
            for line in lines:
                console.print(escape(f"  {line}"), style="dim", highlight=False)
        console.print()


if __name__ == "__main__":
    sys.exit(main())
