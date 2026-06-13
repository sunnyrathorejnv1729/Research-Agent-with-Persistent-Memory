"""
Tool schemas passed to the Anthropic API.
Three tools: web_search, fetch_url, memory_search.
"""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information on a topic. "
            "Use this when the memory store has no relevant results or the "
            "query requires fresh data. Returns a list of search result snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch and extract the main text content from a specific URL. "
            "Use this to get detailed information from a web page found via web_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch content from.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 4000).",
                    "default": 4000,
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search the persistent semantic memory store for previously researched "
            "information. ALWAYS call this first before web_search to avoid redundant "
            "fetches. Returns relevant stored results with similarity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search in memory.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Max number of memory results to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]
