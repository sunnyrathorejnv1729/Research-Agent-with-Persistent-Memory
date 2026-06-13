"""
Concrete implementations of the three agent tools:
  - web_search   — DuckDuckGo via requests (no API key needed)
  - fetch_url    — BeautifulSoup page scraper
  - memory_search — delegates to ResearchMemory
"""

import json
import logging
import time
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 10


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def web_search(query: str, num_results: int = 5) -> dict:
    """
    Search DuckDuckGo HTML interface and return structured snippets.
    Falls back to a concise error dict on failure.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        results = []
        for result in soup.select(".result")[:num_results]:
            title_tag = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")
            url_tag = result.select_one(".result__url")

            if not title_tag:
                continue

            results.append(
                {
                    "title": title_tag.get_text(strip=True),
                    "url": title_tag.get("href", url_tag.get_text(strip=True) if url_tag else ""),
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                }
            )

        logger.info("web_search '%s' → %d results", query, len(results))
        return {"query": query, "results": results, "count": len(results)}

    except Exception as exc:
        logger.error("web_search error: %s", exc)
        return {"query": query, "results": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_chars: int = 4000) -> dict:
    """Fetch a URL and return cleaned main-body text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Prefer <article> or <main>, fall back to <body>
        body = soup.find("article") or soup.find("main") or soup.body
        text = body.get_text(separator="\n", strip=True) if body else soup.get_text()

        # Collapse whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        content = "\n".join(lines)[:max_chars]

        logger.info("fetch_url '%s' → %d chars", url, len(content))
        return {"url": url, "content": content, "chars": len(content)}

    except Exception as exc:
        logger.error("fetch_url error: %s", exc)
        return {"url": url, "content": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# memory_search  (thin wrapper — real logic lives in ResearchMemory)
# ---------------------------------------------------------------------------

def memory_search(memory_store, query: str, n_results: int = 5) -> dict:
    """Query the persistent memory store."""
    hits = memory_store.search(query, n_results=n_results)
    logger.info("memory_search '%s' → %d hits", query, len(hits))
    return {
        "query": query,
        "hits": hits,
        "count": len(hits),
        "memory_hit": len(hits) > 0,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict, memory_store) -> str:
    """Route a tool call to its implementation and return a JSON string result."""
    if name == "web_search":
        result = web_search(
            query=inputs["query"],
            num_results=inputs.get("num_results", 5),
        )
    elif name == "fetch_url":
        result = fetch_url(
            url=inputs["url"],
            max_chars=inputs.get("max_chars", 4000),
        )
    elif name == "memory_search":
        result = memory_search(
            memory_store=memory_store,
            query=inputs["query"],
            n_results=inputs.get("n_results", 5),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, ensure_ascii=False)
