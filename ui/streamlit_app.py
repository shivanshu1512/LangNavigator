"""
Streamlit UI — chat interface for the Docs RAG Assistant.

Features:
- Chat-style input with example-question quick buttons
- Markdown-rendered answers with clickable citation links
- Expandable "Sources and scores" panel showing retrieval details
- Per-stage timing display (retrieve, rerank, generate)
- Refusals shown clearly as info messages, not errors
- Cold-start spinner that polls /health before the first query
- Footer with GitHub link and eval results

The API address comes from st.secrets["API_URL"], set in Streamlit Cloud.
"""

import time

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    API_URL = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    API_URL = "http://localhost:8000"

EXAMPLE_QUESTIONS = [
    "How do I persist state between graph runs?",
    "What is a checkpointer in LangGraph?",
    "How do I add human-in-the-loop to my graph?",
    "What's the difference between StateGraph and MessageGraph?",
    "How do I use tool calling with LangGraph?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LangNavigator — LangGraph Assistant",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Overall dark, premium feel */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
        margin-top: -0.5rem;
    }

    /* Example question buttons */
    .example-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: rgba(102, 126, 234, 0.12);
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #a8b2d1;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .example-btn:hover {
        background: rgba(102, 126, 234, 0.25);
        border-color: rgba(102, 126, 234, 0.6);
        color: #ccd6f6;
    }

    /* Citation links */
    .citation-link {
        color: #667eea;
        text-decoration: none;
        font-weight: 600;
    }
    .citation-link:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    /* Source card in expander */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .source-card h4 {
        color: #ccd6f6;
        margin: 0 0 0.4rem;
        font-size: 0.95rem;
    }
    .source-card .score {
        color: #64ffda;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .source-card .snippet {
        color: #8892b0;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    /* Timing bar */
    .timing-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 0.6rem;
        margin-top: 0.5rem;
        color: #5a6785;
        font-size: 0.78rem;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .timing-bar span {
        color: #64ffda;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #5a6785;
        font-size: 0.8rem;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
    }

    /* Refusal styling */
    .refusal-msg {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 183, 77, 0.08);
        border: 1px solid rgba(255, 183, 77, 0.2);
        border-radius: 12px;
        color: #ffb74d;
        font-size: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API or Direct Pipeline Dispatch
# ---------------------------------------------------------------------------

def check_health() -> bool:
    """Check if the external API is alive. Returns True if healthy."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def get_backend_mode() -> str:
    """Determine whether to use remote HTTP API or embedded Python pipeline."""
    if "backend_mode" not in st.session_state:
        if check_health():
            st.session_state.backend_mode = "api"
        else:
            # Standalone mode: run pipeline directly in-process
            st.session_state.backend_mode = "direct"
    return st.session_state.backend_mode


def query_api(question: str) -> dict | None:
    """Send a question to the API and return the response."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            st.warning("⚠️ Rate limit exceeded. Please wait a moment and try again.")
            return None
        elif resp.status_code == 422:
            detail = resp.json().get("detail", "Invalid input")
            st.warning(f"⚠️ {detail}")
            return None
        else:
            st.error(f"❌ API error ({resp.status_code})")
            return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The server may be under heavy load.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API server.")
        return None


def query_local(question: str) -> dict | None:
    """Run the pipeline directly in Python (standalone Streamlit deployment)."""
    try:
        from docs_rag.pipeline import answer
        result = answer(question=question)
        return {
            "answer": result.answer,
            "refused": result.refused,
            "citations": [
                {
                    "n": c.n,
                    "breadcrumb": c.breadcrumb,
                    "url": c.url,
                    "snippet": c.snippet,
                    "rerank_score": c.rerank_score,
                    "fused_rank": c.fused_rank,
                }
                for c in result.citations
            ],
            "timings_ms": {
                "retrieve": result.timings.retrieve_ms,
                "rerank": result.timings.rerank_ms,
                "generate": result.timings.generate_ms,
                "total": result.timings.total_ms,
            },
            "config": result.config,
            "model": result.model,
            "request_id": result.request_id,
        }
    except Exception as e:
        st.error(f"❌ Error generating answer: {e}")
        return None


def query(question: str) -> dict | None:
    """Dispatches query to API or direct pipeline based on mode."""
    if get_backend_mode() == "api":
        return query_api(question)
    return query_local(question)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🧭 LangNavigator</h1>
    <p>Ask anything about LangGraph — answers grounded in the official docs with citations</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example question buttons
if not st.session_state.messages:
    st.markdown("##### 💡 Try an example question:")
    cols = st.columns(len(EXAMPLE_QUESTIONS[:3]))
    for i, q in enumerate(EXAMPLE_QUESTIONS[:3]):
        with cols[i]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

    cols2 = st.columns(len(EXAMPLE_QUESTIONS[3:]))
    for i, q in enumerate(EXAMPLE_QUESTIONS[3:]):
        with cols2[i]:
            if st.button(q, key=f"example_{i+3}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Render the answer
            data = msg.get("data", {})
            if data.get("refused"):
                st.markdown(f'<div class="refusal-msg">🔍 {data.get("answer", "Not found in docs")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            # Sources expander
            citations = data.get("citations", [])
            if citations:
                with st.expander(f"📄 Sources and scores ({len(citations)} sources)", expanded=False):
                    for c in citations:
                        score_text = f"Rerank score: {c['rerank_score']:.3f}" if c.get("rerank_score") is not None else f"Fused rank: #{c.get('fused_rank', '?')}"
                        st.markdown(f"""
<div class="source-card">
    <h4>[{c['n']}] {c['breadcrumb']}</h4>
    <div class="score">{score_text}</div>
    <div class="snippet">{c['snippet']}</div>
    <a class="citation-link" href="{c['url']}" target="_blank">→ View in docs</a>
</div>
""", unsafe_allow_html=True)

            # Timings
            timings = data.get("timings_ms", {})
            if timings:
                st.markdown(f"""
<div class="timing-bar">
    🔍 Retrieve: <span>{timings.get('retrieve', 0):.0f}ms</span>
    ⚖️ Rerank: <span>{timings.get('rerank', 0):.0f}ms</span>
    🤖 Generate: <span>{timings.get('generate', 0):.0f}ms</span>
    ⏱️ Total: <span>{timings.get('total', 0):.0f}ms</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about LangGraph..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Query via API or local pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching docs and generating answer..."):
            result = query(question)

        if result:
            if result.get("refused"):
                st.markdown(f'<div class="refusal-msg">🔍 {result["answer"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(result["answer"])

            # Sources
            citations = result.get("citations", [])
            if citations:
                with st.expander(f"📄 Sources and scores ({len(citations)} sources)", expanded=False):
                    for c in citations:
                        score_text = f"Rerank score: {c['rerank_score']:.3f}" if c.get("rerank_score") is not None else f"Fused rank: #{c.get('fused_rank', '?')}"
                        st.markdown(f"""
<div class="source-card">
    <h4>[{c['n']}] {c['breadcrumb']}</h4>
    <div class="score">{score_text}</div>
    <div class="snippet">{c['snippet']}</div>
    <a class="citation-link" href="{c['url']}" target="_blank">→ View in docs</a>
</div>
""", unsafe_allow_html=True)

            # Timings
            timings = result.get("timings_ms", {})
            if timings:
                st.markdown(f"""
<div class="timing-bar">
    🔍 Retrieve: <span>{timings.get('retrieve', 0):.0f}ms</span>
    ⚖️ Rerank: <span>{timings.get('rerank', 0):.0f}ms</span>
    🤖 Generate: <span>{timings.get('generate', 0):.0f}ms</span>
    ⏱️ Total: <span>{timings.get('total', 0):.0f}ms</span>
</div>
""", unsafe_allow_html=True)

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "data": result,
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Something went wrong.",
                "data": {},
            })

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="footer">
    Built by <a href="https://github.com/shivanshu1512" target="_blank">Shivanshu Shukla</a> ·
    <a href="https://github.com/shivanshu1512/LangNavigator" target="_blank">GitHub</a> ·
    Questions are logged anonymously to improve the system
</div>
""", unsafe_allow_html=True)
