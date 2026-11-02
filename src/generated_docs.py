"""Keep generated tables inside hand-written documents in step with their JSON reports.

A document marks the generated part with a start comment naming the command that writes
it, and the shared end comment. Everything outside the markers is left untouched, so the
narrative can be edited by hand while the numbers are always rewritten by code.
"""

from __future__ import annotations

import re
from pathlib import Path

GENERATED_END = "<!-- END GENERATED -->"


def replace_generated_block(doc: Path, start: str, content: str) -> None:
    text = doc.read_text()
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(GENERATED_END), re.S)
    if not pattern.search(text):
        raise ValueError(f"{doc} has no generated block markers")
    doc.write_text(pattern.sub(lambda _: f"{start}\n{content}\n{GENERATED_END}", text))
