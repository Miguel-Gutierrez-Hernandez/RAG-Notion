"""Grimorio: interfaz web para consultar tus notas de Dragones y Mazmorras.

Uso:  streamlit run app.py
"""
from datetime import datetime

import requests
import streamlit as st

from config import settings
from src.citations import cited_numbers
from src.embeddings import EmbeddingError, OllamaEmbedder
from src.hardware_check import detect_hardware
from src.hybrid_retriever import HybridRetriever
from src.images import collect_answer_images, resolve_images
from src.indexer import Indexer, IndexerError
from src.llm import LLMError, OllamaChat
from src.model_selector import select_chat_model
from src.notion_loader import NotionError
from src.pipeline import run_index, run_sync
from src.qa_chain import QAChain
from src.sync_state import load_state

AVATARS = {"user": ":material/person:", "assistant": ":material/auto_stories:"}
ALL_PAGES = "Todo el grimorio"

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root { --gold:#c8a15a; --ember:#b5472a; --ink:#0f0d0b; --panel:#17130f; --parchment:#e8dfd0; }

/* Tipografía: serif clásica (si no hay internet, cae a Georgia) */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp [data-testid="stMarkdownContainer"] {
  font-family: 'EB Garamond', Georgia, 'Times New Roman', serif;
}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li { font-size: 1.12rem; line-height: 1.6; }

/* Limpieza: fuera menús y adornos de Streamlit */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 46rem; }

/* Cabecera */
.title { font-family:'Cinzel', Georgia, serif; font-size:2.1rem; font-weight:600;
         letter-spacing:.18em; text-transform:uppercase; color:var(--gold) !important; margin:0; }
.tagline { font-style:italic; color:#d6c9b0 !important; margin:.15rem 0 0 0; font-size:1.05rem; }
.rule { height:1px; margin:1rem 0 1.4rem 0;
        background:linear-gradient(90deg, var(--gold), rgba(200,161,90,0)); }
.hint { text-align:center; font-style:italic; color:#c9bca3 !important; margin:4rem 0 2rem 0; font-size:1.25rem; }

/* Mensajes: sin burbujas, una línea dorada para el asistente */
[data-testid="stChatMessage"] { background:transparent; border:none; padding:.6rem 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  border-left:2px solid rgba(200,161,90,.35); padding-left:1rem; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { background:transparent; }

/* Entrada de chat */
[data-testid="stChatInput"] { border:1px solid rgba(200,161,90,.35); border-radius:6px; background:var(--panel); }
[data-testid="stChatInput"]:focus-within { border-color:var(--gold); box-shadow:0 0 0 1px rgba(200,161,90,.35); }
[data-testid="stBottom"] > div { background:var(--ink); }

/* Barra lateral */
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid rgba(200,161,90,.15); }
[data-testid="stSidebar"] .brand { font-family:'Cinzel', Georgia, serif; letter-spacing:.2em;
  text-transform:uppercase; color:var(--gold); font-size:1.05rem; margin-bottom:.2rem; }

/* Botones: contorno dorado, sobrios */
.stButton button { background:transparent; border:1px solid rgba(200,161,90,.45); color:var(--parchment);
  border-radius:4px; letter-spacing:.06em; transition:all .15s ease; }
.stButton button:hover { border-color:var(--gold); color:var(--gold); background:rgba(200,161,90,.06); }
.stButton button[kind="primary"] { background:var(--gold); color:var(--ink); border-color:var(--gold); font-weight:600; }
.stButton button[kind="primary"]:hover { background:#d8b36c; color:var(--ink); }

/* Detalles */
[data-testid="stExpander"] { border:1px solid rgba(200,161,90,.2); border-radius:4px; background:transparent; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#f2e9d6 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#d6c9b0 !important; opacity:1 !important; }
/* Etiquetas de los controles ("Buscar solo en", "Modelo", sliders...) */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p { color:#f2e9d6 !important; }
/* Streamlit oculta (opacity 0) el mínimo y el máximo de los sliders hasta pasar el ratón */
[data-testid="stSliderTickBar"] { opacity:1 !important; }
[data-testid="stSliderTickBar"] p, [data-testid="stSliderTickBar"] div { color:#d6c9b0 !important; opacity:1 !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color:var(--gold) !important; }
/* Iconos de ayuda ("?") bien visibles */
[data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] svg { color:#e0bd78 !important; opacity:1 !important; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipIcon"] svg * { stroke:#e0bd78 !important; }
[data-testid="stTooltipContent"] { border:1px solid rgba(200,161,90,.5); border-radius:4px;
  background:#17130f !important; color:#f2e9d6 !important; }
[data-testid="stTooltipContent"] p { color:#f2e9d6 !important; }
[data-testid="stSlider"] { margin-bottom:.9rem; }
[data-testid="stChatMessageAvatarUser"] span, [data-testid="stChatMessageAvatarAssistant"] span { color:var(--gold); }
a { color:var(--gold) !important; text-decoration:none; border-bottom:1px dotted rgba(200,161,90,.5); }
a:hover { color:#e0bd78 !important; border-bottom-style:solid; }
hr { border-color:rgba(200,161,90,.15); }

/* ---- Paleta forzada: la app se ve igual sea cual sea el tema que tenga Streamlit ---- */
.stApp { color-scheme: dark; }
.stApp, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
  background: var(--ink) !important; color: var(--parchment); }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
  background: var(--panel) !important; }
.stApp [data-testid="stMarkdownContainer"] :is(p, li, strong, em, span, h1, h2, h3, h4, td, th, blockquote) {
  color: var(--parchment); }
[data-testid="stExpander"], [data-testid="stExpander"] details { background: transparent !important; }
/* El texto de los botones hereda el color del propio botón (oscuro sobre el dorado) */
.stButton button p { color: inherit !important; }
.stButton button[kind="primary"], .stButton button[kind="primary"]:hover { color: var(--ink) !important; }
[data-testid="stExpander"] summary { background: var(--panel) !important; }
[data-testid="stExpander"] summary:hover { background: rgba(200,161,90,.10) !important; }
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code { background: var(--panel) !important; }
.stApp :not(pre) > code { background: #241d15 !important; color: #e0bd78 !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {
  background: var(--panel) !important; color: var(--parchment) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #a99e8c !important; opacity: 1; }
[data-testid="stSelectbox"] [role="group"] { background: var(--ink) !important; border-color: rgba(200,161,90,.3) !important; }
[data-testid="stSelectbox"] [role="group"] * { color: var(--parchment) !important; }
[role="listbox"], div:has(> [role="listbox"]) { background: var(--panel) !important; }
[role="option"] { background: var(--panel) !important; color: var(--parchment) !important; }
[role="option"][aria-selected="true"], [role="option"]:hover { background: rgba(200,161,90,.16) !important; }
</style>
"""

# La barra lateral arranca abierta solo si aún no hay nada sincronizado.
_state = load_state(settings.sync_state_file)
st.set_page_config(page_title="Grimorio", page_icon=":material/auto_stories:", layout="centered",
                   initial_sidebar_state="collapsed" if _state["pages"] else "expanded")
st.markdown(STYLE, unsafe_allow_html=True)


# ---------- Recursos en caché ----------
@st.cache_resource
def get_default_model():
    return select_chat_model(detect_hardware(), settings.chat_model)


@st.cache_resource
def get_indexer():
    return Indexer(settings.vectorstore_dir,
                   OllamaEmbedder(settings.ollama_host, settings.embedding_model),
                   settings.chunk_size, settings.chunk_overlap)


@st.cache_resource
def get_retriever():
    return HybridRetriever(get_indexer(), settings.top_k)


@st.cache_data(ttl=10)
def installed_models(host: str):
    try:
        r = requests.get(f"{host}/api/tags", timeout=2)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except requests.RequestException:
        return None


def has_model(names: list[str], wanted: str) -> bool:
    return any(n == wanted or n.split(":")[0] == wanted for n in names)


def pages_under(pages: dict, selected_path: str) -> list[str]:
    """IDs de la página elegida y de todas sus subpáginas."""
    return [pid for pid, p in pages.items()
            if p["path"] == selected_path or p["path"].startswith(selected_path + " > ")]


def clean(label: str) -> str:
    return label.replace("[", "\\[").replace("]", "\\]")


def hit_label(h) -> str:
    return f"{h.path} > {h.section}" if h.section else h.path


def render_images(images: list[dict]) -> None:
    """Imágenes de las notas que acompañan a la respuesta (en cuadrícula de 3)."""
    shown = [i for i in images if i.get("path")]
    if shown:
        cols = st.columns(min(3, len(shown)))
        for n, im in enumerate(shown):
            caption = f"{im['caption']} · {im['label']}" if im.get("caption") else im["label"]
            with cols[n % len(cols)]:
                st.image(im["path"], caption=caption)
    for im in images:
        if not im.get("path"):
            st.caption(f"{im.get('caption') or 'Imagen'}: no está descargada. "
                       f"[Verla en Notion]({im['url']})")


def render_sources(hits: list[dict], text: str) -> None:
    if not hits:
        return
    nums = cited_numbers(text, len(hits))
    st.markdown("**Fuentes citadas**" if nums else "**Fragmentos consultados**")
    for n in (nums or range(1, len(hits) + 1)):
        h = hits[n - 1]
        link = f"[{clean(h['label'])}]({h['url']})" if h["url"] else clean(h["label"])
        st.markdown(f"- **[{n}]** {link}")
    with st.expander(f"Ver los {len(hits)} fragmentos recuperados"):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {clean(h['label'])}** · relevancia {h['score']}")
            st.markdown(h["content"])
            for im in h.get("images", []):
                if im.get("path"):
                    st.image(im["path"], width=170, caption=im.get("caption") or None)
            if i < len(hits):
                st.divider()


# ---------- Estado ----------
state = load_state(settings.sync_state_file)
pages = state["pages"]
default_choice = get_default_model()

try:
    indexer = get_indexer()
except IndexerError as e:
    st.error(f"{e}\n\nEjecuta en la terminal: `python3 -m scripts.index_notion --rebuild`")
    st.stop()
retriever = get_retriever()


def do_sync(full: bool = False, rebuild: bool = False) -> None:
    if problems := settings.validate_notion():
        st.sidebar.error("\n\n".join(problems) + "\n\nConfigura tu archivo `.env` y reinicia la app.")
        return
    with st.status("Actualizando el grimorio...", expanded=True) as status:
        try:
            sync = run_sync(settings, full=full or rebuild, progress=st.write)
            status.update(label="Inscribiendo los cambios...")
            idx = run_index(settings, indexer, rebuild=rebuild, progress=st.write)
        except (NotionError, EmbeddingError, IndexerError) as e:
            status.update(label="Algo salió mal", state="error")
            st.error(str(e))
            return
        status.update(label="Grimorio al día", state="complete", expanded=False)
    st.session_state.flash = (f"{sync.total} páginas · {sync.changed} nuevas o modificadas · "
                              f"{sync.deleted} eliminadas · {idx.chunks} fragmentos")
    st.rerun()


# ---------- Barra lateral ----------
with st.sidebar:
    st.markdown('<div class="brand">Grimorio</div>', unsafe_allow_html=True)
    if flash := st.session_state.pop("flash", None):
        st.success(flash)

    synced = state.get("synced_at")
    st.caption(f"{len(pages)} páginas · {indexer.count()} fragmentos"
               + (f" · {datetime.fromisoformat(synced):%d/%m %H:%M}" if synced else ""))
    pending = len(state["pending"]["changed"]) + len(state["pending"]["deleted"])
    if pending:
        st.warning(f"{pending} páginas pendientes de indexar.")

    options = [ALL_PAGES] + sorted({p["path"] for p in pages.values()})
    selected = st.selectbox("Buscar solo en", options, help="Incluye sus subpáginas.")
    page_ids = None if selected == ALL_PAGES else pages_under(pages, selected)

    if st.button("Actualizar grimorio", use_container_width=True, type="primary"):
        do_sync()
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    models = installed_models(settings.ollama_host)
    chat_candidates = [default_choice.chat_model]
    if models:
        chat_candidates += [m for m in models if m not in chat_candidates
                            and "embed" not in m and not m.startswith("bge")]

    with st.expander("Ajustes"):
        chat_model = st.selectbox("Modelo", chat_candidates,
                                  help=f"Recomendado: {default_choice.chat_model}")
        top_k = st.slider("Fragmentos a consultar", 1, 10, settings.top_k)
        temperature = st.slider("Creatividad", 0.0, 1.0, 0.2, 0.05,
                                help="Más baja = más fiel a tus notas.")
        rewrite = st.checkbox("Entender preguntas de seguimiento", value=True,
                              help="Reescribe la pregunta usando la conversación. "
                                   "Desactívalo para respuestas más rápidas.")
        st.divider()
        if st.button("Resincronizar todo", use_container_width=True):
            do_sync(full=True)
        if st.button("Reindexar desde cero", use_container_width=True,
                     help="Úsalo si cambiaste el modelo de embeddings."):
            do_sync(rebuild=True)

    # Avisos solo cuando hay un problema real
    if models is None:
        st.error(f"Ollama no responde en {settings.ollama_host}.\n\n`brew services start ollama`")
    else:
        for needed in (chat_model, settings.embedding_model):
            if not has_model(models, needed):
                st.warning(f"Falta el modelo `{needed}`: `ollama pull {needed}`")


# ---------- Chat ----------
st.markdown('<p class="title">Grimorio</p>'
            '<p class="tagline">Crónicas, reglas y mazmorras</p>'
            '<div class="rule"></div>', unsafe_allow_html=True)

if indexer.count() == 0:
    st.info("El grimorio está vacío. Abre el panel lateral y pulsa **Actualizar grimorio**.")
    st.stop()
if page_ids:
    st.caption(f"Buscando solo en: {selected}")

st.session_state.setdefault("messages", [])
if not st.session_state.messages:
    st.markdown('<p class="hint">¿Qué deseas saber, aventurero?</p>', unsafe_allow_html=True)
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(m["content"])
        render_images(m.get("images", []))
        render_sources(m.get("hits", []), m["content"])

if prompt := st.chat_input("Pregunta al grimorio…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        retriever.top_k = top_k
        llm = OllamaChat(settings.ollama_host, chat_model, temperature=temperature)
        chain = QAChain(retriever, llm, rewrite=rewrite)
        placeholder = st.empty()
        placeholder.caption("🕯️ Consultando los pergaminos…")
        try:
            answer = chain.ask(prompt, history, page_ids)

            def tokens():
                first = True
                for token in answer.tokens:
                    if first:
                        placeholder.empty()
                        first = False
                    yield token

            text = st.write_stream(tokens())
        except (EmbeddingError, LLMError, IndexerError) as e:
            placeholder.empty()
            st.error(str(e))
        else:
            hits = [{"label": hit_label(h), "url": h.url, "score": h.score,
                     "content": h.content,
                     "images": resolve_images(h.images, settings.images_dir)}
                    for h in answer.hits]
            images = collect_answer_images(answer.hits, text, prompt, settings.images_dir)
            render_images(images)
            render_sources(hits, text)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "hits": hits, "images": images})
