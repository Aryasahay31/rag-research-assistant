"""
Streamlit Chat UI — RAG Research Assistant frontend.

Run with: streamlit run ui/streamlit_app.py
"""

import streamlit as st
import requests

# --- Configuration ---
API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="magnifying_glass_tilted_left",
    layout="wide",
)

# --- Custom CSS for clean, classic look ---
st.markdown("""
<style>
    /* Clean sans-serif throughout */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Reduce sidebar width */
    [data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 320px;
    }

    /* Clean header */
    .main-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 2px;
        letter-spacing: -0.3px;
    }
    .main-subtitle {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 12px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 10px 14px;
        flex: 1;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #888;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Stage breakdown */
    .stage-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid #f0f0f0;
    }
    .stage-name { color: #444; }
    .stage-time { color: #888; font-variant-numeric: tabular-nums; }

    /* Chunk cards */
    .chunk-card {
        background: #fafafa;
        border-left: 3px solid #ddd;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #333;
    }
    .chunk-meta {
        font-size: 0.72rem;
        color: #999;
        margin-bottom: 4px;
    }

    /* Sidebar section headers */
    .sidebar-section {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #888;
        margin-top: 16px;
        margin-bottom: 8px;
        font-weight: 600;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-ok { background: #22c55e; }
    .status-err { background: #ef4444; }

    /* Hide default streamlit header padding */
    .block-container { padding-top: 2rem; }

    /* Cleaner buttons */
    .stButton > button {
        font-size: 0.82rem;
        font-weight: 500;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-default"


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="PDF, DOCX, or TXT",
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{API_URL}/upload", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(result["message"])
                else:
                    st.error(response.json().get("detail", "Upload failed"))
            except requests.ConnectionError:
                st.error("Cannot connect to API.")

    st.markdown('<div class="sidebar-section">Settings</div>', unsafe_allow_html=True)

    use_hyde = st.toggle(
        "HyDE Query Expansion",
        value=False,
        help="Generates a hypothetical answer first to improve retrieval for vague queries. Adds latency.",
    )

    show_metrics = st.toggle(
        "Show Evaluation Metrics",
        value=True,
        help="Show latency breakdown, groundedness score, and source chunks.",
    )

    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear All Documents", use_container_width=True):
        try:
            response = requests.delete(f"{API_URL}/documents")
            if response.status_code == 200:
                st.success("All documents cleared.")
                st.session_state.messages = []
            else:
                st.error("Failed to clear documents.")
        except requests.ConnectionError:
            st.error("Cannot connect to API.")

    # Status
    st.markdown('<div class="sidebar-section">Status</div>', unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        redis_ok = health.get("redis_connected", False)
        api_html = '<span class="status-dot status-ok"></span>API'
        redis_html = f'<span class="status-dot {"status-ok" if redis_ok else "status-err"}"></span>Redis'
        st.markdown(f"{api_html} &nbsp;&nbsp; {redis_html}", unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-dot status-err"></span>API offline', unsafe_allow_html=True)


# --- Main Area ---
st.markdown('<div class="main-title">RAG Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Hybrid search with cross-encoder reranking</div>', unsafe_allow_html=True)


def render_metrics(result):
    """Render clean evaluation metrics."""
    latency = result.get("latency", {})
    groundedness = result.get("groundedness_score", 0)
    chunks_used = result.get("chunks_used", [])

    total_ms = latency.get("total_ms", 0)

    # Metric cards
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value">{total_ms / 1000:.1f}s</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Groundedness</div>
            <div class="metric-value">{groundedness:.0%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sources Used</div>
            <div class="metric-value">{len(chunks_used)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Latency breakdown
    with st.expander("Latency Breakdown"):
        stages = []
        if latency.get("cache_check_ms"):
            stages.append(("Cache Check", latency["cache_check_ms"]))
        if latency.get("hyde_ms"):
            stages.append(("HyDE Transform", latency["hyde_ms"]))
        if latency.get("retrieval_ms"):
            stages.append(("Hybrid Retrieval", latency["retrieval_ms"]))
        if latency.get("reranking_ms"):
            stages.append(("Reranking", latency["reranking_ms"]))
        if latency.get("generation_ms"):
            stages.append(("LLM Generation", latency["generation_ms"]))

        for name, ms in stages:
            pct = (ms / total_ms * 100) if total_ms > 0 else 0
            st.markdown(
                f'<div class="stage-row">'
                f'<span class="stage-name">{name}</span>'
                f'<span class="stage-time">{ms:.0f}ms ({pct:.1f}%)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Source chunks
    if chunks_used:
        with st.expander(f"Source Chunks ({len(chunks_used)})"):
            for i, chunk in enumerate(chunks_used, 1):
                score = chunk.get("rerank_score", 0)
                source = chunk.get("source", "Unknown")
                page = chunk.get("page", "N/A")
                text = chunk.get("text", "")

                st.markdown(
                    f'<div class="chunk-meta">'
                    f'{source} / Page {page} — relevance: {score:.4f}'
                    f'</div>'
                    f'<div class="chunk-card">{text}</div>',
                    unsafe_allow_html=True,
                )


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_metrics and message.get("metrics"):
            render_metrics(message["metrics"])


# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "use_hyde": use_hyde,
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]

                    if result.get("cached"):
                        st.caption("Served from cache")

                    st.markdown(answer)

                    if result.get("sources"):
                        sources_str = ", ".join(result["sources"])
                        st.caption(f"Sources: {sources_str}")

                    if show_metrics:
                        render_metrics(result)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metrics": result if show_metrics else None,
                    })
                else:
                    error = response.json().get("detail", "Unknown error")
                    st.error(error)

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to the API. Make sure the backend is running: "
                    "uvicorn app.main:app --reload --port 8000"
                )