"""The About tab's documents exist in both languages and the diagram converts cleanly."""

from __future__ import annotations

import re

import pytest

from frontend.about import (
    DIAGRAM_CSS,
    DOCUMENTS,
    LABEL_WIDTH,
    mermaid_to_dot,
    read_document,
)


@pytest.mark.parametrize("name", list(DOCUMENTS))
@pytest.mark.parametrize("language", ["en", "fr"])
def test_every_document_is_present(name, language):
    assert read_document(name, language)


def test_customer_notice_says_the_same_in_both_languages():
    english, french = read_document("notice", "en"), read_document("notice", "fr")
    assert "never decides alone" in english and "ne décide jamais seul" in french
    assert "privacy@borealmarche.ca" in english and "privacy@borealmarche.ca" in french
    rights = [len(re.findall(r"^- ", text, re.M)) for text in (english, french)]
    assert rights[0] == rights[1] == 4


def test_architecture_diagram_keeps_every_edge_of_the_mermaid_source():
    source = read_document("architecture")
    dot = mermaid_to_dot(source)
    mermaid = re.search(r"```mermaid(.*?)```", source, re.S).group(1)
    assert dot.startswith("digraph") and "rankdir=TB" in dot
    assert dot.count(" -> ") == mermaid.count("-->")
    assert 'label="FastAPI"' in dot


def test_text_without_a_flowchart_has_no_diagram():
    assert mermaid_to_dot("# Title\n\nNo diagram here.") is None


def test_diagram_groups_every_box_and_wraps_long_labels():
    source = read_document("architecture")
    dot = mermaid_to_dot(source)
    mermaid = re.search(r"```mermaid(.*?)```", source, re.S).group(1)
    titles = re.findall(r"subgraph \w+\[([^\]]+)\]", mermaid)
    assert len(titles) == dot.count("subgraph cluster_") == 6
    assert all(f'label="{title}"' in dot for title in titles)
    boxes = re.findall(r'^\s*\w+ \[label="([^"]*)"', dot, re.M)
    # Every box sits inside a group: none is left floating outside at two-space indent.
    assert not re.findall(r'^  \w+ \[label="', dot, re.M)
    assert max(len(part) for box in boxes for part in box.split("\\n")) <= LABEL_WIDTH


def test_diagram_text_size_survives_the_page_stylesheet():
    assert "stGraphVizChart" in DIAGRAM_CSS and "!important" in DIAGRAM_CSS
