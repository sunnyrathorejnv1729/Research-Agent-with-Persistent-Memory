# 🔬 Research Agent with Persistent Memory

An agentic research assistant powered by **Anthropic's Tool-Use API**, with **persistent semantic memory** backed by **ChromaDB** and **all-MiniLM-L6-v2** embeddings. Built with a clean **Streamlit** UI and a full **evaluation pipeline**.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Autonomous tool orchestration** | Claude dynamically selects between `memory_search`, `web_search`, and `fetch_url` |
| **Persistent semantic memory** | ChromaDB + sentence-transformer embeddings survive across sessions |
| **Memory-first workflow** | Agent always checks memory before hitting the web, eliminating redundant fetches |
| **Evaluation pipeline** | Measures memory hit rate, keyword coverage, latency, and tool efficiency |
| **Streamlit UI** | Three-tab interface: Research · History · Evaluation Dashboard |
| **CLI support** | `python main.py "your query"` for quick terminal use |

---

## 🏗 Architecture

```
research_agent/
├── app.py                  # Streamlit UI (3 tabs)
├── agent.py                # ResearchAgent — Anthropic tool-use orchestrator
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example
│
├── memory/
│   └── memory_store.py     # ChromaDB + all-MiniLM-L6-v2 semantic memory
│
├── tools/
│   ├── tool_schemas.py     # Anthropic tool JSON schemas
│   └── tool_executor.py    # web_search · fetch_url · memory_search implementations
│
├── evaluation/
│   └── eval_pipeline.py    # Benchmark runner + metric aggregation
│
└── utils/
    └── helpers.py          # Logging, env validation
```

### Agent Workflow

```
User Query
    │
    ▼
memory_search ──► HIT  ──► Synthesise answer from memory
    │
    ▼ MISS
web_search
    │
    ▼
fetch_url (top 1-2 pages)
    │
    ▼
Store in ChromaDB
    │
    ▼
Synthesise answer
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/research-agent.git
cd research-agent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Or use the CLI

```bash
# Single query
python main.py "What is retrieval-augmented generation?"

# Run evaluation benchmark
python main.py --eval

# Clear memory store
python main.py --clear
```

---

## 📊 Evaluation Metrics

The evaluation pipeline (`evaluation/eval_pipeline.py`) measures:

| Metric | Description |
|---|---|
| **memory_hit_rate** | Fraction of queries answered purely from memory |
| **repeat_memory_hit_rate** | Hit rate on repeated queries (target: 100%) |
| **keyword_coverage** | Fraction of expected keywords found in the answer |
| **latency_seconds** | Wall-clock time per research turn |
| **avg_tool_calls** | Average number of tool invocations |
| **tool_efficiency** | 1.0 if memory hit, else 0.0 |

Benchmark results are saved to `evaluation/eval_results.json` and visualised in the Streamlit Evaluation tab.

---

## 🔧 Configuration

All settings can be tuned via `.env`:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-6
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
MEMORY_SEARCH_THRESHOLD=0.75    # cosine similarity threshold for memory hits
MAX_MEMORY_RESULTS=5
```

---

## 🛠 Tech Stack

- **[Anthropic Python SDK](https://github.com/anthropic/anthropic-sdk-python)** — Tool-use API for agentic orchestration
- **[ChromaDB](https://www.trychroma.com/)** — Persistent local vector database
- **[sentence-transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` for fast local embeddings
- **[LangChain](https://python.langchain.com/)** — Utility integrations
- **[Streamlit](https://streamlit.io/)** — Interactive web UI
- **[Plotly](https://plotly.com/python/)** — Evaluation charts
- **[BeautifulSoup4](https://beautiful-soup-4.readthedocs.io/)** — HTML scraping for `fetch_url`

---

## 📈 Performance

On the included benchmark (7 queries, 2 repeated):

| Metric | Result |
|---|---|
| Repeat query memory hit rate | **100%** |
| Average latency | **< 3s** |
| Average keyword coverage | **≥ 85%** |

---

## 📄 License

MIT
