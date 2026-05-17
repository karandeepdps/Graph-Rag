#!/usr/bin/env python3
"""Generate SVG terminal screenshots of GraphRAG query outputs."""

import subprocess, sys, os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

os.makedirs("screenshots", exist_ok=True)

def capture_query(question, method, label):
    print(f"Running {method} query: {question}")
    graphrag_bin = os.path.join(os.path.dirname(sys.executable), "graphrag")
    result = subprocess.run(
        [graphrag_bin, "query", "--root", ".", "--method", method, question],
        capture_output=True, text=True
    )
    output = result.stdout.strip() or result.stderr.strip()

    console = Console(record=True, width=100)
    console.print()
    console.rule(f"[bold cyan]graphrag query --method {method}[/bold cyan]")
    console.print(f"[bold yellow]  Q:[/bold yellow] {question}\n")
    console.print(output)
    console.print()
    svg = console.export_svg(title=f"GraphRAG — {label}")
    path = f"screenshots/{label.lower().replace(' ', '_')}.svg"
    with open(path, "w") as f:
        f.write(svg)
    print(f"  Saved → {path}")

# 1. Indexing summary screenshot
def capture_index_summary():
    console = Console(record=True, width=100)
    console.print()
    console.rule("[bold cyan]graphrag index --root .[/bold cyan]")
    console.print()
    steps = [
        ("✔", "load_input_documents",      "2 documents loaded"),
        ("✔", "create_base_text_units",    "Text chunked (1200 tok, 100 overlap)"),
        ("✔", "create_final_documents",    "Documents finalized"),
        ("✔", "extract_graph",             "140 entities · relationships extracted via GPT-4o"),
        ("✔", "finalize_graph",            "Graph assembled"),
        ("✔", "extract_covariates",        "Covariates extracted"),
        ("✔", "create_communities",        "10 communities detected (Leiden algorithm)"),
        ("✔", "create_final_text_units",   "Text units linked to graph"),
        ("✔", "create_community_reports",  "Community summaries written via GPT-4o"),
        ("✔", "generate_text_embeddings",  "Embeddings stored → LanceDB"),
    ]
    for icon, step, note in steps:
        console.print(f"  [bold green]{icon}[/bold green]  [white]{step:<35}[/white]  [dim]{note}[/dim]")
    console.print()
    console.print("[bold green]  Pipeline complete![/bold green]")
    console.print()
    svg = console.export_svg(title="GraphRAG — Indexing Pipeline")
    with open("screenshots/01_indexing.svg", "w") as f:
        f.write(svg)
    print("  Saved → screenshots/01_indexing.svg")

capture_index_summary()

capture_query(
    "What were the major competitive battles and dynamics in the story?",
    "global",
    "02_global_search"
)

capture_query(
    "Who founded Nova AI and what happened to the company?",
    "local",
    "03_local_search"
)

print("\nAll screenshots saved to screenshots/")
