"""
GraphRAG Visualizer
-------------------
Reads the indexed knowledge graph and produces an interactive HTML file
you can open in any web browser — no extra software needed.

Usage:
    python visualize.py                    # uses latest output/ run
    python visualize.py --max-nodes 80     # limit to the 80 highest-degree nodes
    python visualize.py --community 0      # show only community 0
    python visualize.py --output my.html   # custom output filename

What you'll see:
    • Each NODE is an entity (person, company, event, place, ...)
      - Color encodes entity TYPE (blue=org, red=person, green=geo, ...)
      - Size encodes how many connections the entity has (more connections = bigger)
    • Each EDGE is a relationship between two entities
    • COMMUNITIES appear as distinct color-coded clusters
    • Hover over any node or edge to read its description
    • Drag nodes to rearrange, scroll to zoom, click to highlight neighbors
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyvis.network import Network


# ── Visual settings ──────────────────────────────────────────────────────────

# One color per entity type (add more if your settings.yml has custom types)
TYPE_COLORS = {
    "organization": "#4ECDC4",   # teal
    "person":       "#FF6B6B",   # coral red
    "geo":          "#95E1D3",   # mint green
    "event":        "#F7DC6F",   # golden yellow
    "technology":   "#BB8FCE",   # soft purple
    "product":      "#F0A500",   # amber
    "unknown":      "#CCCCCC",   # grey fallback
}

# One color per community (cycles if there are more than 10 communities)
COMMUNITY_PALETTE = [
    "#E63946", "#2A9D8F", "#E9C46A", "#264653", "#F4A261",
    "#A8DADC", "#457B9D", "#1D3557", "#F72585", "#7209B7",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_output_dir() -> Path:
    output_root = Path("output")
    if not output_root.exists():
        print(
            "\nERROR: No 'output/' directory found.\n"
            "       Run the indexing pipeline first:\n"
            "       graphrag index --root .\n"
        )
        sys.exit(1)
    return output_root


def load_graph_data(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (nodes_df, edges_df) from GraphRAG v0.5 flat parquet output."""
    nodes_path = output_dir / "entities.parquet"
    edges_path = output_dir / "relationships.parquet"
    communities_path = output_dir / "communities.parquet"

    if not nodes_path.exists():
        print(f"\nERROR: {nodes_path} not found. Did indexing complete?\n")
        sys.exit(1)

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)

    # Join community membership onto entities (use top-level communities, level=0)
    if communities_path.exists():
        communities = pd.read_parquet(communities_path)
        top = communities[communities["level"] == 0][["community", "entity_ids"]]
        mapping = (
            top.explode("entity_ids")
            .rename(columns={"entity_ids": "id"})
            .drop_duplicates("id")
            [["id", "community"]]
        )
        nodes = nodes.merge(mapping, on="id", how="left")

    return nodes, edges


def node_color(row: pd.Series, use_community: bool) -> str:
    """Pick a color based on community membership (if available) or entity type."""
    if use_community and "community" in row and pd.notna(row["community"]):
        idx = int(row["community"]) % len(COMMUNITY_PALETTE)
        return COMMUNITY_PALETTE[idx]
    entity_type = str(row.get("type", "unknown")).lower()
    return TYPE_COLORS.get(entity_type, TYPE_COLORS["unknown"])


# ── Main ──────────────────────────────────────────────────────────────────────

def build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    max_nodes: int | None,
    community_filter: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter and trim the graph according to CLI options."""

    # Filter to a single community if requested
    if community_filter is not None:
        if "community" not in nodes.columns:
            print("WARNING: No community column found — ignoring --community filter.")
        else:
            keep = nodes[nodes["community"] == community_filter]["title"].tolist()
            nodes = nodes[nodes["community"] == community_filter]
            edges = edges[
                edges["source"].isin(keep) & edges["target"].isin(keep)
            ]
            print(f"  Filtered to community {community_filter}: {len(nodes)} nodes, {len(edges)} edges")

    # degree column already provided by GraphRAG v0.5 entities.parquet
    if "degree" not in nodes.columns:
        all_endpoints = pd.concat([edges["source"], edges["target"]])
        deg = all_endpoints.value_counts().rename("degree")
        nodes = nodes.join(deg, on="title", how="left")
    nodes["degree"] = nodes["degree"].fillna(1).astype(int)

    if max_nodes and len(nodes) > max_nodes:
        nodes = nodes.nlargest(max_nodes, "degree")
        keep = set(nodes["title"])
        edges = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
        print(f"  Trimmed to top {max_nodes} nodes by degree: {len(edges)} edges remain")

    return nodes, edges


def render(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_path: str,
    color_by: str,
) -> None:
    use_community_color = color_by == "community"

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",      # dark navy background — easier to read
        font_color="#ffffff",
        directed=False,
        notebook=False,
    )

    # Physics: a force-directed layout that clusters related nodes naturally
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.003,
          "springLength": 150,
          "springConstant": 0.08,
          "damping": 0.9
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 200 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "smooth": { "type": "continuous" },
        "color": { "color": "#555577", "highlight": "#ffffff" },
        "font": { "size": 10, "color": "#aaaacc" },
        "arrows": { "to": { "enabled": false } }
      }
    }
    """)

    # Add nodes
    for _, row in nodes.iterrows():
        title = str(row.get("title", "?"))
        description = str(row.get("description", "No description"))
        entity_type = str(row.get("type", "unknown")).lower()
        degree = int(row.get("degree", 1))
        community = row.get("community")
        color = node_color(row, use_community_color)

        community_label = f" | Community {int(community)}" if pd.notna(community) else ""
        tooltip = (
            f"<b>{title}</b><br>"
            f"Type: {entity_type.title()}{community_label}<br>"
            f"Connections: {degree}<br><br>"
            f"{description[:300]}{'...' if len(description) > 300 else ''}"
        )

        net.add_node(
            title,
            label=title,
            title=tooltip,          # tooltip shown on hover
            color=color,
            size=8 + min(degree * 3, 40),   # bigger = more connected, capped at 48px
            font={"size": max(10, min(degree + 9, 20))},
            borderWidth=2,
            borderWidthSelected=4,
        )

    # Add edges
    for _, row in edges.iterrows():
        src = str(row.get("source", ""))
        tgt = str(row.get("target", ""))
        desc = str(row.get("description", ""))
        weight = float(row.get("weight", 1.0))

        if not src or not tgt:
            continue

        net.add_edge(
            src, tgt,
            title=desc[:200],           # edge tooltip
            width=max(1, weight * 2),   # thicker = stronger relationship
        )

    net.save_graph(output_path)


def print_legend(color_by: str) -> None:
    print("\n  LEGEND")
    if color_by == "community":
        print("  Node color = community (cluster of related entities)")
        for i, color in enumerate(COMMUNITY_PALETTE[:6]):
            print(f"    Community {i}: {color}")
        print("    ...")
    else:
        print("  Node color = entity type:")
        for entity_type, color in TYPE_COLORS.items():
            if entity_type != "unknown":
                print(f"    {color}  {entity_type.title()}")
    print("  Node size  = number of connections (bigger = more connected)")
    print("  Edge width = relationship strength (thicker = stronger)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize your GraphRAG knowledge graph as an interactive HTML file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tips:
  • In the browser: scroll to zoom, drag to pan, drag nodes to rearrange
  • Hover over a node or edge to read its description
  • Use --max-nodes 50 if the graph is too dense to read
  • Use --community 0 to zoom in on one cluster of related entities
  • Use --color type to color by entity type instead of community
        """,
    )
    parser.add_argument(
        "--output", "-o",
        default="graph.html",
        help="Output HTML filename (default: graph.html)",
    )
    parser.add_argument(
        "--max-nodes", "-n",
        type=int,
        default=None,
        help="Keep only the N most-connected nodes (useful for large graphs)",
    )
    parser.add_argument(
        "--community", "-c",
        type=int,
        default=None,
        help="Show only nodes belonging to this community ID",
    )
    parser.add_argument(
        "--color",
        choices=["community", "type"],
        default="community",
        help="Color nodes by community (default) or by entity type",
    )
    args = parser.parse_args()

    print("\n  GraphRAG Visualizer")
    print("  " + "─" * 40)

    artifacts = find_output_dir()
    print(f"  Source: {artifacts}")

    nodes, edges = load_graph_data(artifacts)
    print(f"  Loaded: {len(nodes)} nodes, {len(edges)} edges")

    nodes, edges = build_graph(nodes, edges, args.max_nodes, args.community)
    print_legend(args.color)

    print(f"\n  Rendering → {args.output} ...", end=" ", flush=True)
    render(nodes, edges, args.output, args.color)
    print("done")

    print(f"\n  Open in your browser:")
    print(f"    open {args.output}          (macOS)")
    print(f"    start {args.output}         (Windows)")
    print(f"    xdg-open {args.output}      (Linux)")
    print()


if __name__ == "__main__":
    main()
