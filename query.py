#!/usr/bin/env python3
"""Convenience wrapper around graphrag query CLI."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Query the GraphRAG index")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument(
        "--method",
        choices=["local", "global"],
        default="global",
        help="local = specific entity lookup, global = broad thematic questions (default: global)",
    )
    parser.add_argument("--root", default=".", help="GraphRAG root directory (default: .)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "graphrag.query",
        "--root", args.root,
        "--method", args.method,
        args.question,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
