# notion-rag

Asistente local (RAG) que responde preguntas sobre tus páginas de Notion, 100 % en tu equipo
(salvo la descarga desde Notion). Modelos con Ollama, índice en ChromaDB, interfaz con Streamlit.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # rellena NOTION_TOKEN y NOTION_ROOT_PAGE_ID

brew install ollama && brew services start ollama
ollama pull bge-m3              # embeddings
ollama pull qwen2.5:3b          # chat (ver recomendación en el chequeo)

python3 -m scripts.check_system # comprueba hardware, Ollama y credenciales
```

Notion: https://www.notion.so/profile/integrations -> nueva integración interna -> copia el token.
En tu página raíz: `···` -> *Connections* -> añade la integración. El ID son los 32 caracteres finales de la URL.

## Uso diario

```bash
streamlit run app.py            # interfaz web (botón "Sincronizar con Notion" incluido)
```

También por terminal:

```bash
python3 -m scripts.sync_notion            # descarga lo que cambió (--full: todo)
python3 -m scripts.index_notion           # indexa lo pendiente (--rebuild: desde cero)
python3 -m scripts.ask_cli --debug        # chat por terminal
python3 -m scripts.search "pregunta"      # solo recuperación, sin LLM
```

## Estructura

```
app.py                 interfaz Streamlit
config.py              configuración (.env)
src/
  hardware_check.py    detecta CPU/RAM/GPU
  model_selector.py    elige el modelo según el hardware
  notion_loader.py     API de Notion, recorrido recursivo
  notion_markdown.py   bloques -> Markdown
  sync_state.py        qué cambió / qué falta indexar
  pipeline.py          sincronizar + indexar (usado por la app)
  chunker.py           división en fragmentos con contexto
  embeddings.py        embeddings con Ollama
  indexer.py           ChromaDB
  retriever.py         búsqueda vectorial
  hybrid_retriever.py  vectorial + BM25 (RRF)
  llm.py               chat con Ollama (streaming)
  prompts.py           prompts en español
  qa_chain.py          recuperación + LLM + historial
  citations.py         fuentes citadas
scripts/               utilidades de terminal
tests/                 pruebas offline:  python3 -m tests.test_qa  (etc.)
```

## Tests (no necesitan Notion ni Ollama)

```bash
for t in test_markdown test_sync test_chunker test_index test_qa test_pipeline; do python3 -m tests.$t; done
```

## Problemas frecuentes

- **"Sin acceso a ... (404)"**: la página raíz no está compartida con la integración.
- **"Modelo ... no encontrado"**: ejecuta el `ollama pull` que indica el mensaje.
- **Cambiaste `EMBEDDING_MODEL`**: reindexa con `python3 -m scripts.index_notion --rebuild`.
- **Respuestas lentas**: normal en CPU. Prueba `qwen2.5:1.5b` o desactiva "Entender preguntas de seguimiento".
