"""Estado de sincronización: qué páginas existen y qué falta por indexar."""
import json
from pathlib import Path

EMPTY = {"pages": {}, "pending": {"changed": [], "deleted": []}}


def load_state(path: Path) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state.setdefault("pages", {})
        state.setdefault("pending", {"changed": [], "deleted": []})
        return state
    return json.loads(json.dumps(EMPTY))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def mark_pending(state: dict, changed: list[str], deleted: list[str]) -> None:
    """Acumula cambios pendientes de indexar (la fase 3 los consumirá)."""
    pending = state["pending"]
    ch = [i for i in pending["changed"] if i not in deleted] + \
         [i for i in changed if i not in pending["changed"]]
    de = [i for i in pending["deleted"] if i not in changed] + \
         [i for i in deleted if i not in pending["deleted"]]
    pending["changed"], pending["deleted"] = ch, de
