"""
AI Research Assistant — Multi-Source RAG with Dual Groq LLM Pipeline.
Streamlit interface: document upload, semantic search, and cited chat answers.
"""

import streamlit as st

try:
    import html as html_lib
    import uuid
    import streamlit.components.v1 as components

    from utils.document_processor import process_uploaded_files, DocumentProcessingError
    from utils.chunking import chunk_documents
    from utils.embeddings import EmbeddingGenerator
    from utils.vector_store import VectorStore
    from utils.memory import ConversationMemory
    from utils.llm_handler import GroqLLMHandler, LLMError
except ImportError as e:
    st.set_page_config(page_title="Researcher", layout="wide")
    st.error(
        f"A required package is missing: **{e.name}**.\n\n"
        "Please run `pip install -r requirements.txt` in your project's virtual "
        "environment, then restart the app."
    )
    st.stop()

st.set_page_config(page_title="Researcher", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def new_conversation() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New chat",
        "messages": [],
        "memory": ConversationMemory(),
    }


defaults = {
    "theme": "dark",
    "sidebar_expanded": True,
    "api_key_verified": False,
    "groq_api_key": "",
    "vector_store": None,
    "embedder": None,
    "processed_files": [],
    "top_k": 4,
    "conversations": [],
    "active_conversation_id": None,
    "search_query": "",
    "pending_uploads": None,
    "upload_feedback": None
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

if not st.session_state.conversations:
    convo = new_conversation()
    st.session_state.conversations.append(convo)
    st.session_state.active_conversation_id = convo["id"]


def get_active_conversation() -> dict:
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_conversation_id:
            return c
    convo = new_conversation()
    st.session_state.conversations.append(convo)
    st.session_state.active_conversation_id = convo["id"]
    return convo


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    dark = st.session_state.theme == "dark"

    bg = "#212121" if dark else "#ffffff"
    bg_elevated = "#2a2a2a" if dark else "#f7f7f7"
    bg_input = "#2f2f2f" if dark else "#f0f0f0"
    bg_bubble = "#343434" if dark else "#eeeeee"
    text = "#ececec" if dark else "#1a1a1a"
    text_muted = "#9b9b9b" if dark else "#6e6e6e"
    border = "#3d3d3d" if dark else "#e2e2e2"
    accent = "#d97757"

    sidebar_width = "260px" if st.session_state.sidebar_expanded else "60px"

    st.markdown(f"""
    <style>
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; height: 0; }}
        div[data-testid="stToolbar"] {{ display: none; }}
        div[data-testid="stDecoration"] {{ display: none; }}
        header {{ background-color: transparent !important; }}

        /* Hide native sidebar collapse control — every version's markup, by testid AND aria-label */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[aria-label*="sidebar" i],
        button[aria-label*="collapse" i] {{
            display: none !important;
        }}

        /* Override native collapse transform/margin so our own state fully controls visibility */
        section[data-testid="stSidebar"] {{
            transform: none !important;
            margin-left: 0px !important;
            visibility: visible !important;
            position: relative !important;
            background-color: {bg_elevated};
            border-right: 1px solid {border};
            width: {sidebar_width} !important;
            min-width: {sidebar_width} !important;
            max-width: {sidebar_width} !important;
        }}
        div[data-testid="stSidebarUserContent"] {{
            visibility: visible !important;
        }}
        section[data-testid="stSidebar"] * {{ color: {text} !important; }}

        .block-container {{ padding-top: 0.8rem; max-width: 780px; }}
        .stApp {{ background-color: {bg}; color: {text}; }}

        input, textarea {{
            background-color: {bg_input} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }}
        input:focus, textarea:focus {{
            border-color: {accent} !important;
            box-shadow: none !important;
            outline: none !important;
        }}
        input::placeholder, textarea::placeholder {{ color: {text_muted} !important; }}

        .stButton button {{
            background-color: {bg_input};
            color: {text};
            border: 1px solid {border};
            border-radius: 8px;
            font-weight: 400;
        }}
        .stButton button:hover {{
            border-color: {accent};
            color: {accent};
            background-color: {bg_input};
        }}
        .stButton button p {{ color: inherit !important; }}

        /* Fix invisible icons (popover trigger, buttons) across themes */
        .stButton button svg,
        div[data-testid="stPopover"] button svg,
        [data-testid="stIconMaterial"] {{
            fill: {text} !important;
            color: {text} !important;
        }}
        div[data-testid="stPopover"] button {{
            background-color: {bg_input} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
        }}

        div[data-testid="stChatInput"] {{
            background-color: {bg_input};
            border: 1px solid {border};
            border-radius: 14px;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"]:focus-within {{
            border-color: {accent} !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}

        .app-title {{ font-size: 1.5rem; font-weight: 600; color: {text}; margin-bottom: 0.2rem; }}
        .app-subtitle {{ color: {text_muted}; font-size: 0.9rem; margin-bottom: 0.6rem; }}
        .section-label {{
            color: {text_muted}; font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.6rem 0 0.4rem 0;
        }}
        .doc-chip {{
            display: inline-block; background-color: {bg_input};
            border: 1px solid {border}; border-radius: 6px;
            padding: 3px 9px; margin: 2px 4px 2px 0;
            font-size: 0.78rem; color: {text_muted};
        }}
        .status-line {{ color: {text_muted}; font-size: 0.85rem; font-style: italic; }}

        .user-bubble {{
            background-color: {bg_bubble};
            border-radius: 14px;
            padding: 8px 14px;
            margin: 2px 0 8px 0;
            display: inline-block;
            max-width: 90%;
            color: {text};
            white-space: pre-wrap;
        }}
        .assistant-block {{ margin-bottom: 12px; }}
    </style>
    """, unsafe_allow_html=True)


def copy_button(text: str, key: str):
    safe_text = html_lib.escape(text).replace("`", "\\`").replace("\n", "\\n")
    components.html(f"""
        <button id="copy-btn-{key}" onclick="
            navigator.clipboard.writeText(`{safe_text}`);
            const btn = document.getElementById('copy-btn-{key}');
            btn.innerText = 'Copied';
            setTimeout(() => {{ btn.innerText = 'Copy'; }}, 1500);
        "
            style="background:transparent;border:1px solid #555;color:#999;
            border-radius:6px;padding:3px 12px;font-size:12px;cursor:pointer;
            font-family:inherit;">
            Copy
        </button>
    """, height=32)

# ---------------------------------------------------------------------------
# API key gate — blocking modal, validated before entry
# ---------------------------------------------------------------------------
@st.dialog("Connect Groq API")
def api_key_dialog():
    st.markdown("Enter your Groq API key to begin.")
    key_input = st.text_input("Groq API Key", type="password", key="dialog_api_key")
    submitted = st.button("Continue", key="dialog_submit")

    if submitted:
        if not key_input.strip():
            st.error("Please enter a key.")
            return
        with st.spinner("Verifying key..."):
            try:
                handler = GroqLLMHandler(api_key=key_input.strip())
                handler.validate_key()
            except LLMError as e:
                st.error(str(e))
                return
        st.session_state.groq_api_key = key_input.strip()
        st.session_state.api_key_verified = True
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:

        # ---- Collapsed: icon-only rail ----
        if not st.session_state.sidebar_expanded:
            st.markdown(f"""
            <style>
                section[data-testid="stSidebar"] button {{
                    border: none !important;
                    background-color: transparent !important;
                    padding: 4px !important;
                    min-height: 34px !important;
                    width: 40px !important;
                    font-size: 1.05rem !important;
                }}
                section[data-testid="stSidebar"] button:hover {{
                    background-color: {"#2f2f2f" if st.session_state.theme == "dark" else "#f0f0f0"} !important;
                    border-radius: 8px !important;
                }}
                section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                    gap: 0.2rem !important;
                }}
                section[data-testid="stSidebar"] .element-container {{
                    margin-bottom: 0 !important;
                }}
            </style>
            """, unsafe_allow_html=True)

            if st.button("▶", key="sidebar_toggle_collapsed"):
                st.session_state.sidebar_expanded = True
                st.rerun()

            if st.button("+", key="rail_new_chat"):
                convo = new_conversation()
                st.session_state.conversations.insert(0, convo)
                st.session_state.active_conversation_id = convo["id"]
                st.rerun()

            with st.popover("💬"):
                st.markdown("**Chats**")
                search_val = st.text_input(
                    "Search chats", placeholder="Search your chats...",
                    label_visibility="collapsed", key="rail_chat_search",
                )
                query = search_val.strip().lower()
                for convo in st.session_state.conversations:
                    if query:
                        title_match = query in convo["title"].lower()
                        content_match = any(query in m["content"].lower() for m in convo["messages"])
                        if not (title_match or content_match):
                            continue
                    label = convo["title"][:32] + ("..." if len(convo["title"]) > 32 else "")
                    if st.button(label, key=f"rail_convo_{convo['id']}", use_container_width=True):
                        st.session_state.active_conversation_id = convo["id"]
                        st.rerun()

            with st.popover("🔍"):
                st.markdown("**Search**")
                st.text_input(
                    "Search", placeholder="Search your chats...",
                    label_visibility="collapsed", key="rail_search_only",
                )

            theme_icon = "☀" if st.session_state.theme == "dark" else "🌙"
            if st.button(theme_icon, key="rail_theme_toggle"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()

            return

        # ---- Expanded sidebar ----
        if st.button("◀", key="sidebar_toggle_expanded"):
            st.session_state.sidebar_expanded = False
            st.rerun()

        st.markdown('<div class="app-title" style="font-size:1.2rem;">Researcher</div>', unsafe_allow_html=True)

        if st.button("+  New chat", key="new_chat", use_container_width=True):
            convo = new_conversation()
            st.session_state.conversations.insert(0, convo)
            st.session_state.active_conversation_id = convo["id"]
            st.rerun()

        with st.expander("Upload documents", expanded=False):
            sidebar_uploaded = st.file_uploader(
                "Upload documents", type=["pdf", "docx", "txt"],
                accept_multiple_files=True, label_visibility="collapsed", key="sidebar_uploader",
            )
            if sidebar_uploaded:
                new_files = [f for f in sidebar_uploaded if f.name not in st.session_state.processed_files]
                if new_files:
                    st.session_state.pending_uploads = sidebar_uploaded
                    st.rerun()

        st.session_state.search_query = st.text_input(
            "Search chats", value=st.session_state.search_query,
            placeholder="Search chats...", label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">Chats</div>', unsafe_allow_html=True)

        query = st.session_state.search_query.strip().lower()
        for convo in st.session_state.conversations:
            if query:
                title_match = query in convo["title"].lower()
                content_match = any(query in m["content"].lower() for m in convo["messages"])
                if not (title_match or content_match):
                    continue
            label = convo["title"][:32] + ("..." if len(convo["title"]) > 32 else "")
            if st.button(label, key=f"convo_{convo['id']}", use_container_width=True):
                st.session_state.active_conversation_id = convo["id"]
                st.rerun()

        st.divider()
        st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
        if st.session_state.processed_files:
            chips = "".join(f'<span class="doc-chip">{html_lib.escape(f)}</span>'
                             for f in st.session_state.processed_files)
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-line">No documents indexed yet</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)

        st.session_state.top_k = st.slider("Top-K retrieval", min_value=2, max_value=10,
                                             value=st.session_state.top_k)

        theme_choice = st.radio(
            "Interface", options=["Dark", "Light"],
            index=0 if st.session_state.theme == "dark" else 1,
            horizontal=True, key="theme_radio",
        )
        new_theme = "dark" if theme_choice == "Dark" else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        if st.button("Change API key", key="change_api_key", use_container_width=True):
            st.session_state.api_key_verified = False
            st.rerun()

        with st.expander("Groq API Key", expanded=False):
            masked = (st.session_state.groq_api_key[:4] + "••••••••") if st.session_state.groq_api_key else "Not set"
            st.caption(f"Current key: {masked}")
            new_key = st.text_input("Update API key", type="password", key="sidebar_api_key_edit")
            if st.button("Save key", key="save_sidebar_key"):
                if new_key.strip():
                    try:
                        handler = GroqLLMHandler(api_key=new_key.strip())
                        handler.validate_key()
                        st.session_state.groq_api_key = new_key.strip()
                        st.success("API key updated.")
                    except LLMError as e:
                        st.error(str(e))
                else:
                    st.error("Please enter a key.")


# ---------------------------------------------------------------------------
# Document ingestion — staged status feedback
# ---------------------------------------------------------------------------
def handle_uploads(uploaded_files):
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if not new_files:
        return {"errors": [], "indexed_count": 0, "already_indexed": True}

    with st.status("Indexing documents...", expanded=True) as status:
        st.write("Extracting text...")
        documents, errors = process_uploaded_files(new_files)

        if not documents:
            status.update(label="Indexing failed", state="error")
            return {"errors": errors, "indexed_count": 0, "already_indexed": False}

        st.write("Chunking content...")
        chunks = chunk_documents(documents)
        if not chunks:
            status.update(label="No usable content found", state="error")
            return {"errors": errors + ["No usable content found in the uploaded file(s)."],
                     "indexed_count": 0, "already_indexed": False}

        st.write("Generating embeddings...")
        if st.session_state.embedder is None:
            st.session_state.embedder = EmbeddingGenerator()
        texts = [c["text"] for c in chunks]
        embeddings = st.session_state.embedder.encode(texts)

        st.write("Building vector index...")
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(dimension=embeddings.shape[1])
        st.session_state.vector_store.add(embeddings, chunks)

        indexed_names = {c["metadata"]["source"] for c in chunks}
        st.session_state.processed_files.extend(
            [f.name for f in new_files if f.name in indexed_names]
        )

        status.update(label=f"Indexed {len(indexed_names)} file(s)", state="complete")

    return {"errors": errors, "indexed_count": len(indexed_names), "already_indexed": False}


# ---------------------------------------------------------------------------
# Chat handling — staged status feedback, document-aware answers
# ---------------------------------------------------------------------------
EXT_KEYWORDS = {"pdf": "pdf", "docx": "docx", "doc": "docx", "word": "docx", "txt": "txt", "text file": "txt"}

import re

GENERIC_WORDS = {
    "the", "a", "an", "is", "of", "in", "on", "for", "to", "and", "what",
    "does", "do", "about", "describe", "tell", "me", "this", "that",
    "project", "document", "doc", "docs", "file", "files", "pdf", "docx",
    "txt", "report", "paper", "task", "uploaded", "upload",
}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in GENERIC_WORDS}


def detect_target_sources(question: str) -> set:
    """Scope retrieval to a specific file if the question's meaningful
    words clearly overlap with that filename's meaningful words."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return set()

    matches = {}
    for fname in st.session_state.processed_files:
        name_without_ext = fname.rsplit(".", 1)[0]
        f_tokens = _tokenize(name_without_ext)
        overlap = q_tokens & f_tokens
        if overlap:
            matches[fname] = len(overlap)

    if not matches:
        return set()

    # Only restrict to a file if it's a clearly stronger match than the rest —
    # avoids accidentally narrowing on a weak coincidental overlap.
    best_score = max(matches.values())
    top_matches = {f for f, score in matches.items() if score == best_score}
    return top_matches

def handle_question(question: str):
    question = question.strip()
    if not question:
        return

    convo = get_active_conversation()
    convo["messages"].append({"role": "user", "content": question, "sources": []})

    if st.session_state.vector_store is None or st.session_state.vector_store.is_empty:
        convo["messages"].append({
            "role": "assistant",
            "content": "Please upload at least one document before asking questions.",
            "sources": [],
        })
        return

    try:
        llm = GroqLLMHandler(api_key=st.session_state.groq_api_key)

        with st.status("Retrieving relevant chunks...", expanded=False) as status:
            query_embedding = st.session_state.embedder.encode([question])[0]
            target_sources = detect_target_sources(question)
            if target_sources:
                retrieved = st.session_state.vector_store.search_filtered(
                    query_embedding, target_sources, top_k=st.session_state.top_k
                )
            else:
                retrieved = st.session_state.vector_store.search(query_embedding, top_k=st.session_state.top_k)
            status.update(label="Summarizing context...")
            summarized_context = llm.summarize_context(
                retrieved, convo["memory"].get_history_text(), st.session_state.processed_files,
            )

            status.update(label="Finalizing answer...")
            answer = llm.generate_answer(question, summarized_context, st.session_state.processed_files)

            if convo["title"] == "New chat":
                status.update(label="Naming conversation...")
                convo["title"] = llm.generate_title(question, answer)

            status.update(label="Done", state="complete")

        convo["memory"].add_turn(question, answer)
        convo["messages"].append({
            "role": "assistant", "content": answer, "sources": retrieved,
            "summarized_context": summarized_context,
        })

    except LLMError as e:
        convo["messages"].append({"role": "assistant", "content": f"Error: {e}", "sources": []})
    except Exception as e:
        convo["messages"].append({"role": "assistant", "content": f"Unexpected error: {e}", "sources": []})


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def render_main():
    convo = get_active_conversation()

    st.markdown('<div class="app-title">Researcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Ask questions across all your uploaded documents.</div>',
                unsafe_allow_html=True)

    if st.session_state.get("pending_uploads"):
        pending = st.session_state.pending_uploads
        st.session_state.pending_uploads = None
        feedback = handle_uploads(pending)
        st.session_state.upload_feedback = feedback
        st.rerun()

    if st.session_state.get("upload_feedback"):
        fb = st.session_state.upload_feedback
        st.session_state.upload_feedback = None
        for err in fb.get("errors", []):
            st.error(err)
        if fb.get("indexed_count", 0) > 0:
            st.success(f"{fb['indexed_count']} file(s) uploaded successfully. You can ask questions now.")
        if fb.get("already_indexed"):
            st.info("These files are already indexed.")

    for i, msg in enumerate(convo["messages"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{html_lib.escape(msg["content"])}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="assistant-block">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            if msg["content"]:
                copy_button(msg["content"], key=f"copy_{convo['id']}_{i}")

            if msg.get("sources"):
                with st.expander("Source references"):
                    for s in msg["sources"]:
                        page = s["metadata"].get("page")
                        label = s["metadata"]["source"] + (f" · page {page}" if page else "")
                        st.markdown(f'<span class="doc-chip">{html_lib.escape(label)} (score: {s["score"]:.3f})</span>', unsafe_allow_html=True)
                        st.caption(s["text"])

            if msg.get("summarized_context"):
                with st.expander("Summarized context"):
                    st.caption(msg["summarized_context"])
            st.markdown('</div>', unsafe_allow_html=True)

    upload_col, input_col = st.columns([1, 11])

    with upload_col:
        with st.popover("+", use_container_width=True):
            uploaded = st.file_uploader(
                "Upload documents", type=["pdf", "docx", "txt"],
                accept_multiple_files=True, label_visibility="collapsed", key="uploader",
            )
            if uploaded:
                new_files = [f for f in uploaded if f.name not in st.session_state.processed_files]
                if new_files:
                    st.session_state.pending_uploads = uploaded
                    st.rerun()

    with input_col:
        question = st.chat_input("Ask a question about your documents...")
        if question:
            handle_question(question)
            st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    inject_css()

    if not st.session_state.api_key_verified:
        api_key_dialog()
        st.stop()

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()