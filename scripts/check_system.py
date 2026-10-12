"""Chequeo previo: hardware, modelo recomendado, Ollama y credenciales Notion.

Uso:  python -m scripts.check_system
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from rich.console import Console
from rich.table import Table

from config import settings
from src.hardware_check import detect_hardware
from src.model_selector import select_chat_model

console = Console()


def check_ollama(required: list[str]) -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{settings.ollama_host}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException:
        return False, required
    installed = {m["name"] for m in r.json().get("models", [])}
    installed_base = installed | {n.split(":")[0] for n in installed}
    missing = [m for m in required if m not in installed and m not in installed_base]
    return True, missing


def main() -> int:
    hw = detect_hardware()
    choice = select_chat_model(hw, settings.chat_model)

    t = Table(title="Hardware detectado")
    t.add_column("Recurso"); t.add_column("Valor")
    t.add_row("Sistema", hw.os)
    t.add_row("CPU", f"{hw.cpu} ({hw.cpu_cores} núcleos)")
    t.add_row("RAM", f"{hw.ram_total_gb} GB (libres: {hw.ram_available_gb} GB)")
    gpu = hw.gpu
    t.add_row("GPU", f"{gpu.name} - {gpu.vram_gb} GB" if hw.has_gpu else "No detectada (modo CPU)")
    console.print(t)

    console.print(f"\n[bold]Modelo de chat:[/bold] {choice.chat_model}  ({choice.reason})")
    console.print(f"[bold]Modelo de embeddings:[/bold] {settings.embedding_model}\n")

    ok = True
    running, missing = check_ollama([choice.chat_model, settings.embedding_model])
    if not running:
        ok = False
        console.print(f"[red]✗ Ollama no responde en {settings.ollama_host}[/red] (instálalo y ejecútalo)")
    elif missing:
        ok = False
        console.print("[yellow]! Faltan modelos en Ollama:[/yellow]")
        for m in missing:
            console.print(f"    ollama pull {m}")
    else:
        console.print("[green]✓ Ollama activo y modelos disponibles[/green]")

    errors = settings.validate_notion()
    if errors:
        ok = False
        for e in errors:
            console.print(f"[red]✗ {e}[/red]")
    else:
        console.print("[green]✓ Credenciales de Notion configuradas[/green]")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
