"""Shared utility helpers."""

import logging
import os
from dotenv import load_dotenv

load_dotenv()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def check_env():
    """Validate required environment variables at startup."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is missing or still set to the placeholder.\n"
            "Copy .env.example → .env and fill in your real API key."
        )
