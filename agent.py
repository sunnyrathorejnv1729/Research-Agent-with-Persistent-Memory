"""
ResearchAgent — autonomous orchestrator that uses the Anthropic tool-use API
to answer research queries through dynamic tool selection:

  memory_search  →  (hit)  return from memory
                 →  (miss) web_search  →  fetch_url  →  store in memory → answer
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from dotenv import load_dotenv

from memory.memory_store import ResearchMemory
from tools.tool_schemas import TOOLS
from tools.tool_executor import execute_tool

load_dotenv()

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an expert research assistant with access to three tools:

1. memory_search  — searches a persistent semantic memory store of past research.
2. web_search     — searches the live web for current information.
3. fetch_url      — retrieves full text from a specific web page.

## Workflow rules (follow in order every time)
1. ALWAYS call memory_search FIRST for the user's query.
2. If memory_search returns relevant hits (memory_hit = true), synthesise a thorough
   answer from those results WITHOUT calling web_search or fetch_url.
3. If memory_search returns no hits, call web_search to find relevant pages.
4. After web_search, call fetch_url on the 1–2 most promising URLs for full content.
5. Synthesise a clear, well-structured answer from the fetched content.
6. Your answer must be factual, cite sources where possible, and be actionable.

Avoid redundant tool calls. Be concise but complete. Use markdown formatting."""


@dataclass
class AgentTrace:
    """Records every step of a single research turn for evaluation."""
    query: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    memory_hit: bool = False
    web_searches: int = 0
    url_fetches: int = 0
    answer: str = ""
    error: Optional[str] = None

    @property
    def latency(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @property
    def tool_efficiency(self) -> float:
        """Ratio of memory hits to total tool calls (higher = more efficient)."""
        total = len(self.tool_calls)
        return round(1.0 if self.memory_hit else 0.0, 2) if total > 0 else 0.0


class ResearchAgent:
    """
    Agentic research assistant powered by Anthropic tool-use.
    Maintains a persistent semantic memory across sessions.
    """

    MAX_ITERATIONS = 10   # guard against infinite loops

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in environment / .env")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._memory = ResearchMemory()
        logger.info("ResearchAgent initialised (model=%s)", CLAUDE_MODEL)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def memory(self) -> ResearchMemory:
        return self._memory

    def research(self, query: str) -> AgentTrace:
        """Run a full research turn and return an AgentTrace with the answer."""
        trace = AgentTrace(query=query)
        messages: list[dict] = [{"role": "user", "content": query}]

        try:
            for _ in range(self.MAX_ITERATIONS):
                response = self._client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )

                # Collect assistant message
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                if response.stop_reason == "end_turn":
                    # Extract final text answer
                    for block in assistant_content:
                        if hasattr(block, "text"):
                            trace.answer = block.text
                    break

                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in assistant_content:
                        if block.type != "tool_use":
                            continue

                        tool_name   = block.name
                        tool_input  = block.input
                        tool_use_id = block.id

                        # Execute the tool
                        result_json = execute_tool(tool_name, tool_input, self._memory)
                        result_data = json.loads(result_json)

                        # Record trace stats
                        trace.tool_calls.append(
                            {"tool": tool_name, "input": tool_input, "result_summary": self._summarise(result_data)}
                        )
                        if tool_name == "memory_search" and result_data.get("memory_hit"):
                            trace.memory_hit = True
                        elif tool_name == "web_search":
                            trace.web_searches += 1
                        elif tool_name == "fetch_url":
                            trace.url_fetches += 1

                        # Persist useful web results to memory automatically
                        if tool_name in ("web_search", "fetch_url") and not result_data.get("error"):
                            content_to_store = self._extract_storable(tool_name, result_data)
                            if content_to_store:
                                self._memory.store(query, content_to_store, source=tool_name)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result_json,
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})

        except Exception as exc:
            logger.exception("Agent error: %s", exc)
            trace.error = str(exc)
            trace.answer = f"An error occurred: {exc}"

        trace.end_time = time.time()
        logger.info(
            "Research done | memory_hit=%s | web=%d | fetch=%d | latency=%.2fs",
            trace.memory_hit, trace.web_searches, trace.url_fetches, trace.latency,
        )
        return trace

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(data: dict) -> str:
        if "hits" in data:
            return f"{data['count']} memory hits"
        if "results" in data:
            return f"{data['count']} web results"
        if "content" in data:
            return f"{data['chars']} chars fetched"
        return "done"

    @staticmethod
    def _extract_storable(tool_name: str, data: dict) -> str:
        """Pull out text worth saving to memory."""
        if tool_name == "web_search":
            snippets = [
                f"{r['title']}: {r['snippet']}" for r in data.get("results", []) if r.get("snippet")
            ]
            return "\n".join(snippets)
        if tool_name == "fetch_url":
            return data.get("content", "")[:3000]
        return ""
