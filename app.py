"""
Research Agent — Streamlit UI
Run:  streamlit run app.py
"""

import json
import os
import sys
import time
import logging
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.helpers import setup_logging, check_env

setup_logging("WARNING")

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ── metric cards ── */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; }

    /* ── answer box ── */
    .answer-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin: 12px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e2e8f0;
    }

    /* ── tool badge ── */
    .tool-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tool-memory  { background: #134e4a; color: #5eead4; border: 1px solid #0d9488; }
    .tool-web     { background: #1e1b4b; color: #a5b4fc; border: 1px solid #6366f1; }
    .tool-fetch   { background: #1c1917; color: #fcd34d; border: 1px solid #d97706; }

    /* ── hit banner ── */
    .memory-hit-banner {
        background: linear-gradient(90deg, #064e3b, #065f46);
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 10px 18px;
        color: #6ee7b7;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .cache-miss-banner {
        background: linear-gradient(90deg, #1e1b4b, #312e81);
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 10px 18px;
        color: #a5b4fc;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* ── input ── */
    .stTextInput > div > div > input {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
    }

    /* ── buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    div[data-testid="stExpander"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state ────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "history" not in st.session_state:
    st.session_state.history = []   # list of AgentTrace
if "init_error" not in st.session_state:
    st.session_state.init_error = None


@st.cache_resource(show_spinner="Loading Research Agent…")
def get_agent():
    """Singleton agent — cached across re-runs."""
    from agent import ResearchAgent
    return ResearchAgent()


def badge(name: str) -> str:
    css = {"memory_search": "tool-memory", "web_search": "tool-web", "fetch_url": "tool-fetch"}
    label = {"memory_search": "🧠 memory", "web_search": "🌐 web", "fetch_url": "📄 fetch"}
    cls   = css.get(name, "tool-web")
    txt   = label.get(name, name)
    return f'<span class="tool-badge {cls}">{txt}</span>'


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Research Agent")
    st.caption("Anthropic Tool-Use · ChromaDB · all-MiniLM-L6-v2")
    st.divider()

    # API key entry
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Paste your key here or set ANTHROPIC_API_KEY in .env",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()

    # Memory stats
    st.markdown("### 🧠 Memory Store")
    try:
        agent = get_agent()
        mem_count = agent.memory.count()
        st.metric("Documents stored", mem_count)
        if st.button("🗑️ Clear memory", use_container_width=True):
            agent.memory.clear()
            st.rerun()
    except Exception as e:
        st.error(f"Agent init failed: {e}")

    st.divider()

    # Eval panel
    st.markdown("### 📊 Evaluation")
    if st.button("▶ Run benchmark", use_container_width=True):
        with st.spinner("Running evaluation benchmark…"):
            try:
                from evaluation.eval_pipeline import run_evaluation
                results = run_evaluation()
                st.session_state["eval_results"] = results
                st.success("Done! See Evaluation tab ↗")
            except Exception as e:
                st.error(f"Eval error: {e}")

    st.divider()
    st.caption("Built with ♥  using Anthropic + ChromaDB + Streamlit")


# ── main layout ──────────────────────────────────────────────────────────────
tab_research, tab_history, tab_eval = st.tabs(["🔍 Research", "📜 History", "📊 Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Research
# ══════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.markdown("## Ask a Research Question")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "query",
            label_visibility="collapsed",
            placeholder="e.g. How do large language models handle long contexts?",
        )
    with col_btn:
        run = st.button("Research →", use_container_width=True)

    # Example queries
    st.markdown("**Quick examples:**")
    ex_cols = st.columns(4)
    examples = [
        "What is retrieval-augmented generation?",
        "How does RLHF work in LLMs?",
        "Explain vector databases",
        "What is the attention mechanism?",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            query = ex
            run = True

    st.divider()

    if run and query.strip():
        try:
            agent = get_agent()
        except Exception as e:
            st.error(f"Could not initialise agent: {e}")
            st.stop()

        with st.spinner("🤖 Researching…"):
            trace = agent.research(query.strip())

        st.session_state.history.append(trace)

        # ── source banner ──
        if trace.memory_hit:
            st.markdown(
                '<div class="memory-hit-banner">⚡ Answered from semantic memory — no web call needed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="cache-miss-banner">🌐 Fetched from web and stored in memory for future queries</div>',
                unsafe_allow_html=True,
            )

        # ── metrics strip ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{trace.latency}s")
        m2.metric("Tool calls", len(trace.tool_calls))
        m3.metric("Web searches", trace.web_searches)
        m4.metric("Pages fetched", trace.url_fetches)

        # ── answer ──
        st.markdown("### 📝 Answer")
        st.markdown(f'<div class="answer-box">{trace.answer}</div>', unsafe_allow_html=True)

        # ── tool trace ──
        with st.expander("🔧 Tool execution trace", expanded=False):
            for step in trace.tool_calls:
                tool = step["tool"]
                inp  = step["input"]
                summ = step["result_summary"]
                st.markdown(
                    f'{badge(tool)} <code style="color:#94a3b8;font-size:0.8rem">'
                    f'{json.dumps(inp)[:120]}</code> → <em style="color:#64748b">{summ}</em>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("## Research History")

    if not st.session_state.history:
        st.info("No research queries yet. Head to the Research tab to get started.")
    else:
        history = list(reversed(st.session_state.history))
        for i, trace in enumerate(history):
            label = f"{'⚡ [MEM]' if trace.memory_hit else '🌐 [WEB]'}  {trace.query[:80]}"
            with st.expander(label, expanded=(i == 0)):
                c1, c2, c3 = st.columns(3)
                c1.metric("Latency", f"{trace.latency}s")
                c2.metric("Tools used", len(trace.tool_calls))
                c3.metric("Memory hit", "✅ Yes" if trace.memory_hit else "❌ No")
                st.markdown(trace.answer)

        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## Evaluation Dashboard")

    results = st.session_state.get("eval_results")

    # Try to load from disk if not in session
    if results is None:
        eval_path = Path("evaluation/eval_results.json")
        if eval_path.exists():
            with open(eval_path) as f:
                results = json.load(f)
            st.session_state["eval_results"] = results

    if results is None:
        st.info("Run the benchmark from the sidebar to see evaluation metrics here.")
    else:
        summary = results["summary"]
        records = results["records"]

        # ── summary metrics ──
        st.markdown("### Summary Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Memory hit rate",  f"{summary['memory_hit_rate']*100:.0f}%")
        c2.metric("Repeat hit rate",  f"{(summary.get('repeat_memory_hit_rate') or 0)*100:.0f}%")
        c3.metric("Avg keyword cov.", f"{summary['avg_keyword_coverage']*100:.0f}%")
        c4.metric("Avg latency",      f"{summary['avg_latency_seconds']:.2f}s")
        c5.metric("Avg tool calls",   summary['avg_tool_calls'])

        st.divider()

        df = pd.DataFrame(records)

        # ── latency bar chart ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Latency per Query")
            fig = px.bar(
                df,
                x="query_id",
                y="latency_seconds",
                color="memory_hit",
                color_discrete_map={True: "#10b981", False: "#6366f1"},
                labels={"latency_seconds": "Latency (s)", "query_id": "Query #", "memory_hit": "Memory hit"},
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", legend_title="Memory Hit")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Keyword Coverage per Query")
            fig2 = px.bar(
                df,
                x="query_id",
                y="keyword_coverage",
                color="memory_hit",
                color_discrete_map={True: "#10b981", False: "#6366f1"},
                labels={"keyword_coverage": "Coverage", "query_id": "Query #", "memory_hit": "Memory hit"},
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", yaxis_tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)

        # ── tool call breakdown ──
        st.markdown("#### Tool Call Breakdown")
        tool_df = df[["query_id", "total_tool_calls", "web_searches", "url_fetches"]].melt(
            id_vars="query_id", var_name="tool", value_name="calls"
        )
        fig3 = px.bar(
            tool_df,
            x="query_id",
            y="calls",
            color="tool",
            barmode="stack",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#6366f1", "#f59e0b"],
            labels={"query_id": "Query #", "calls": "Calls", "tool": "Tool"},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a")
        st.plotly_chart(fig3, use_container_width=True)

        # ── raw table ──
        with st.expander("📋 Raw results table"):
            st.dataframe(
                df[["query_id", "query", "memory_hit", "latency_seconds",
                    "keyword_coverage", "total_tool_calls", "web_searches", "url_fetches", "note"]],
                use_container_width=True,
            )
