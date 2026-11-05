"""The retrieval hold-out set must stay tied to the real corpus."""

import json
import re

import pytest

from src.contracts import PROJECT_ROOT

HOLDOUT_FILE = PROJECT_ROOT / "tests/golden/retrieval_holdout.json"
HOLDOUT = json.loads(HOLDOUT_FILE.read_text())["questions"]
CORPUS = PROJECT_ROOT / "docs/corpus"


def normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def test_ids_are_unique_and_kinds_known():
    ids = [q["id"] for q in HOLDOUT]
    assert len(ids) == len(set(ids))
    assert {q["kind"] for q in HOLDOUT} == {"direct", "indirect"}
    assert {q["language"] for q in HOLDOUT} == {"en", "fr"}


@pytest.mark.parametrize("question", HOLDOUT, ids=lambda q: q["id"])
def test_each_marker_is_quoted_from_its_source_file(question):
    text = normal((CORPUS / question["source"]).read_text())
    assert normal(question["marker"]) in text, "the quoted phrase is not in the policy file"
