#!/usr/bin/env python3
"""
CLI entry point — run a single research query from the terminal.

Usage:
    python main.py "What is retrieval-augmented generation?"
    python main.py --eval        # run full evaluation benchmark
    python main.py --clear       # clear the memory store
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.helpers import setup_logging, check_env

setup_logging("INFO")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Research Agent CLI")
    parser.add_argument("query", nargs="?", help="Research query to answer")
    parser.add_argument("--eval", action="store_true", help="Run evaluation benchmark")
    parser.add_argument("--clear", action="store_true", help="Clear memory store")
    args = parser.parse_args()

    try:
        check_env()
    except EnvironmentError as e:
        print(f"❌  {e}")
        sys.exit(1)

    from agent import ResearchAgent

    agent = ResearchAgent()

    if args.clear:
        agent.memory.clear()
        print("✅  Memory store cleared.")
        return

    if args.eval:
        from evaluation.eval_pipeline import run_evaluation
        run_evaluation()
        return

    if not args.query:
        parser.print_help()
        return

    print(f"\n🔍  Researching: {args.query}\n{'─'*60}")
    trace = agent.research(args.query)

    print(f"\n{'─'*60}")
    print(f"Memory hit  : {'✅ Yes' if trace.memory_hit else '❌ No'}")
    print(f"Latency     : {trace.latency}s")
    print(f"Tool calls  : {len(trace.tool_calls)}")
    print(f"{'─'*60}\n")
    print(trace.answer)


if __name__ == "__main__":
    main()
