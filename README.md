# notion-rag

Asistente local (RAG) que responde preguntas sobre tus páginas de Notion.

## Fase 1: preparación

1. **Entorno**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```
2. **Notion**: https://www.notion.so/profile/integrations -> *New integration* (interna)
   -> copia el token a `NOTION_TOKEN`. En tu página raíz: `···` -> *Connections* -> añade la integración.
   El ID de la página raíz son los 32 caracteres finales de su URL -> `NOTION_ROOT_PAGE_ID`.
3. **Ollama**: instala desde https://ollama.com y descarga los modelos:
   ```bash
   ollama pull bge-m3
   ollama pull qwen2.5:3b     # o el que recomiende el chequeo
   ```
4. **Chequeo**: `python -m scripts.check_system`
