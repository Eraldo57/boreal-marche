"""The undocumented questions must stay undocumented, or they stop testing anything."""

import json
import re

import pytest

from src.contracts import PROJECT_ROOT

NEGATIVES = json.loads((PROJECT_ROOT / "tests/golden/retrieval_negatives.json").read_text())
CORPUS = " ".join(p.read_text().lower() for p in (PROJECT_ROOT / "docs/corpus").glob("*.md"))


@pytest.mark.parametrize("question", NEGATIVES["questions"], ids=lambda q: q["id"])
def test_no_policy_mentions_the_topic(question):
    assert question["absent"], "state which words prove the topic is undocumented"
    for term in question["absent"]:
        assert not re.search(term, CORPUS), f"the corpus now mentions {term!r}"
