"""Documents shown in the About tab, read from the repository (or the frontend image).

Each document has one source of truth in the repository; the interface renders it rather
than keeping its own copy. The architecture diagram is the Mermaid flowchart in
ARCHITECTURE.md, converted to Graphviz because Streamlit draws Graphviz but not Mermaid.
"""

from __future__ import annotations

import re
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


def mermaid_to_dot(markdown: str) -> str | None:
    """Graphviz DOT for the first Mermaid flowchart (A[label] --> B[label] lines)."""
    block = re.search(r"```mermaid\s*\n\s*flowchart\s+(\w+)\n(.*?)```", markdown, re.S)
    if block is None:
        return None
    labels: dict[str, str] = {}
    edges = []
    for line in block.group(2).splitlines():
        if "-->" not in line:
            continue
        left, right = (side.strip() for side in line.split("-->", 1))
        ends = []
        for side in (left, right):
            node = _NODE.fullmatch(side)
            key = node.group(1) if node else side
            if node:
                labels[key] = node.group(2)
            ends.append(key)
        edges.append(tuple(ends))
    direction = "LR" if block.group(1) == "LR" else "TB"
    lines = [f"digraph {{ rankdir={direction}; node [shape=box, style=rounded];"]
    lines += [f'  {key} [label="{label}"];' for key, label in labels.items()]
    lines += [f"  {a} -> {b};" for a, b in edges]
    return "\n".join(lines + ["}"])
