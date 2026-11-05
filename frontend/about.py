"""Documents shown in the About tab, read from the repository (or the frontend image).

Each document has one source of truth in the repository; the interface renders it rather
than keeping its own copy. The architecture diagram is the Mermaid flowchart in
ARCHITECTURE.md, converted to Graphviz because Streamlit draws Graphviz but not Mermaid.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    "notice": "governance/customer_notice_{language}.md",
    "model_card": "MODEL_CARD.md",
    "privacy": "docs/corpus/privacy_notice_{language}.md",
    "provenance": "DATA_CARD.md",
    "architecture": "ARCHITECTURE.md",
}
_NODE = re.compile(r"([A-Za-z0-9_]+)\[([^\]]+)\]")


def read_document(name: str, language: str = "en") -> str | None:
    """The document's text, or None when it is missing from this installation."""
    path = ROOT / DOCUMENTS[name].format(language=language)
    return path.read_text(encoding="utf-8") if path.exists() else None


# Streamlit's stylesheet forces SVG text to 9.6 px, which beats the font size in the diagram.
DIAGRAM_CSS = (
    "<style>"
    '[data-testid="stGraphVizChart"] svg text { font-size: 14px !important; }'
    '[data-testid="stGraphVizChart"] svg g.cluster > text { font-weight: 600; }'
    "</style>"
)
_STYLE = (
    'graph [fontname="Helvetica", nodesep=0.28, ranksep=0.42, newrank=true];\n'
    '  node [shape=box, style="rounded,filled", fillcolor="white", fontcolor="#1b2733",\n'
    '        color="#5b6b7b", fontname="Helvetica", fontsize=14, margin="0.12,0.07"];\n'
    '  edge [color="#7a8794", arrowsize=0.8];'
)
_HUB = 'fillcolor="#2f5d8a", fontcolor="white", color="#2f5d8a"'  # the API, where all paths meet
GROUP_FILLS = ("#eceff3", "#e3eefb", "#e2f4e6", "#fdf0dc", "#e9e4f7", "#fbe6e6")
LABEL_WIDTH = 22
_GROUP = re.compile(r"subgraph\s+([A-Za-z0-9_]+)\[([^\]]+)\]")


def _wrap(label: str) -> str:
    """Break a long label over lines (DOT's \\n) and keep quotes from ending the string."""
    text = label.replace('"', "'")
    return "\\n".join(textwrap.wrap(text, LABEL_WIDTH, break_on_hyphens=False)) or label


def _node(key: str, label: str) -> str:
    return f'{key} [label="{_wrap(label)}"' + (f", {_HUB}" if label == "FastAPI" else "") + "];"


def mermaid_to_dot(markdown: str) -> str | None:
    """Graphviz DOT for the first Mermaid flowchart: A[label] nodes, --> edges, subgraph groups."""
    block = re.search(r"```mermaid\s*\n\s*flowchart\s+(\w+)\n(.*?)```", markdown, re.S)
    if block is None:
        return None
    labels: dict[str, str] = {}
    groups: list[tuple[str, list[str]]] = []
    edges = []
    current: list[str] | None = None
    for raw in block.group(2).splitlines():
        line = raw.strip()
        if group := _GROUP.fullmatch(line):
            current = []
            groups.append((group.group(2), current))
        elif line == "end":
            current = None
        elif "-->" in line:
            left, right = (side.strip() for side in line.split("-->", 1))
            ends = []
            for side in (left, right):
                node = _NODE.fullmatch(side)
                key = node.group(1) if node else side
                if node:
                    labels[key] = node.group(2)
                ends.append(key)
            edges.append(tuple(ends))
        elif node := _NODE.fullmatch(line):
            labels[node.group(1)] = node.group(2)
            if current is not None:
                current.append(node.group(1))
    direction = "LR" if block.group(1) == "LR" else "TB"
    lines = [f"digraph {{ rankdir={direction};\n  {_STYLE}"]
    for position, (title, keys) in enumerate(groups):
        fill = GROUP_FILLS[position % len(GROUP_FILLS)]
        lines.append(f"  subgraph cluster_{position} {{")
        lines.append(f'    label="{title}"; style="rounded,filled"; color="{fill}";')
        lines.append(f'    fillcolor="{fill}"; fontcolor="#1b2733"; fontname="Helvetica";')
        lines.append("    fontsize=14; labeljust=l; margin=12;")
        lines += [f"    {_node(key, labels[key])}" for key in keys]
        lines.append("  }")
    grouped = {key for _, keys in groups for key in keys}
    lines += [f"  {_node(key, label)}" for key, label in labels.items() if key not in grouped]
    lines += [f"  {a} -> {b};" for a, b in edges]
    return "\n".join(lines + ["}"])
