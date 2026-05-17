#!/usr/bin/env python3
"""
GraphRAG Query Helper
---------------------
Run this AFTER indexing your documents with:
    python -m graphrag.index --root .

Usage:
    python query.py "Your question here"
    python query.py "Your question here" --method local
    python query.py "Your question here" --method global

Methods:
    global (default) — Best for broad, thematic questions
                       e.g. "What competitive dynamics shaped the industry?"

    local            — Best for specific facts about named entities
                       e.g. "Who founded Nova AI and what was their background?"

Examples for the included sample story:
    python query.py "Who founded Nova AI?" --method local
    python query.py "What happened during the DocuMind security crisis?" --method local
    python query.py "What were the major competitive battles in the story?" --method global
    python query.py "Compare the strategies of Nova AI and Stellar Systems" --method global
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Query your GraphRAG knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Decision guide — which method to pick:
  Has a named person, company, or event?  →  --method local
  Asking about themes or the big picture?  →  --method global  (default)
  Not sure?                                →  try both and compare!
        """,
    )
    parser.add_argument("question", help="The question you want to ask")
    parser.add_argument(
        "--method",
        choices=["local", "global"],
        default="global",
        help="Search method: 'local' for specific entities, 'global' for broad themes (default: global)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="GraphRAG root directory containing settings.yml (default: current directory)",
    )
    args = parser.parse_args()

    print(f"\n  Method : {args.method.upper()} SEARCH")
    print(f"  Question: {args.question}\n")

    cmd = [
        sys.executable, "-m", "graphrag.query",
        "--root", args.root,
        "--method", args.method,
        args.question,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
