"""The answer-correctness rubric must stay tied to the real corpus and the golden set."""

import json
import re

import pytest

from src.contracts import PROJECT_ROOT

RUBRIC = json.loads((PROJECT_ROOT / "tests/golden/policy_answer_rubric.json").read_text())
GOLDEN = json.loads((PROJECT_ROOT / "tests/golden/golden_questions.json").read_text())["questions"]


def normal(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def test_rubric_covers_every_golden_question_with_an_expected_source():
    expected = {q["id"] for q in GOLDEN if q["expected_source"]}
    assert {item["id"] for item in RUBRIC["questions"]} == expected


@pytest.mark.parametrize("item", RUBRIC["questions"], ids=lambda item: f"Q{item['id']}")
def test_each_fact_is_quoted_from_its_policy_file(item):
    golden = next(q for q in GOLDEN if q["id"] == item["id"])
    assert item["source"] == golden["expected_source"]
    assert (item["marker"] is not None) == item["answerable"], "only answerable facts have a quote"
    if item["marker"]:
        corpus = normal((PROJECT_ROOT / "docs/corpus" / item["source"]).read_text())
        assert normal(item["marker"]) in corpus, "the quoted phrase is not in the policy file"


@pytest.mark.parametrize("item", RUBRIC["questions"], ids=lambda item: f"Q{item['id']}")
def test_each_verdict_is_defined_reasoned_and_tied_to_an_answer(item):
    assert set(item["judgments"]) == {"openai", "ollama"}
    for judgment in item["judgments"].values():
        assert judgment["verdict"] in RUBRIC["verdicts"]
        assert judgment["reason"].strip()
        assert re.fullmatch(r"[0-9a-f]{64}", judgment["reply_sha256"])
