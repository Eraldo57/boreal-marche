"""The About tab's documents exist in both languages and the diagram converts cleanly."""

from __future__ import annotations

import re

import pytest

from frontend.about import DOCUMENTS, mermaid_to_dot, read_document


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
    assert dot.startswith("digraph") and "rankdir=LR" in dot
    assert dot.count(" -> ") == mermaid.count("-->")
    assert 'label="FastAPI"' in dot


def test_text_without_a_flowchart_has_no_diagram():
    assert mermaid_to_dot("# Title\n\nNo diagram here.") is None
