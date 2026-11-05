"""Per-customer explanations: how far each feature of one customer moves their own score.

The models are tiny, fitted on synthetic data generated here, so each effect can be
predicted by hand. They check the method, not the real model's behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from api.routers.predict import MIN_EFFECT, TOP_EFFECTS, _score_with_effects

NAMES = ["strong", "weak", "ignored"]
TYPICAL = {"strong": 0.0, "weak": 0.0, "ignored": 0.0}


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(2000, 3)), columns=NAMES)
    y = (2.5 * frame["strong"] + 0.4 * frame["weak"] + rng.normal(size=2000) > 0).astype(int)
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(frame, y)


def _row(**values: float) -> pd.DataFrame:
    return pd.DataFrame([{**{n: 0.0 for n in NAMES}, **values}])


def test_the_feature_that_drives_the_score_comes_first_with_the_right_sign(model):
    probability, effects = _score_with_effects(model, _row(strong=0.5, weak=1.5), NAMES, TYPICAL)
    assert [e.feature for e in effects][:2] == ["strong", "weak"]
    assert all(e.effect > 0 for e in effects[:2]), "high values here raise the probability"
    assert effects[0].effect > effects[1].effect
    _, low = _score_with_effects(model, _row(strong=-0.5), NAMES, TYPICAL)
    assert low[0].feature == "strong" and low[0].effect < 0


def test_the_probability_is_the_one_a_plain_prediction_gives(model):
    row = _row(strong=1.3, weak=-0.7, ignored=0.4)
    with_effects, _ = _score_with_effects(model, row, NAMES, TYPICAL)
    without, none = _score_with_effects(model, row, NAMES, None)
    assert with_effects == pytest.approx(without) == pytest.approx(model.predict_proba(row)[0, 1])
    assert none == []


def test_a_typical_value_has_no_effect_and_is_not_reported(model):
    _, effects = _score_with_effects(model, _row(strong=0.5), NAMES, TYPICAL)
    assert "ignored" not in [e.feature for e in effects]
    assert all(abs(e.effect) >= MIN_EFFECT for e in effects)


def test_effects_are_the_difference_from_scoring_with_the_typical_value(model):
    row = _row(strong=0.5)
    probability, effects = _score_with_effects(model, row, NAMES, TYPICAL)
    at_typical = model.predict_proba(_row(strong=0.0))[0, 1]
    strong = next(e for e in effects if e.feature == "strong")
    assert strong.effect == pytest.approx(probability - at_typical, abs=1e-4)
    assert strong.value == 0.5 and strong.typical_value == 0.0


def test_only_the_largest_few_are_returned_in_order_of_size(model):
    names = [f"f{i}" for i in range(12)]
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(1500, 12)), columns=names)
    y = (frame.sum(axis=1) + rng.normal(size=1500) > 0).astype(int)
    wide = make_pipeline(StandardScaler(), LogisticRegression()).fit(frame, y)
    # Values that grow with the feature number keep the probability mid-range, so each
    # feature moves it by a different amount and the largest are the last ones.
    row = pd.DataFrame([{n: 0.03 * (i + 1) for i, n in enumerate(names)}])
    _, effects = _score_with_effects(wide, row, names, dict.fromkeys(names, 0.0))
    sizes = [abs(e.effect) for e in effects]
    assert len(effects) == TOP_EFFECTS and sizes == sorted(sizes, reverse=True)
    assert {e.feature for e in effects} <= set(names[-TOP_EFFECTS - 2 :])


def test_missing_typical_values_mean_no_explanation_not_a_failure(model):
    probability, effects = _score_with_effects(model, _row(strong=1.0), NAMES, {"strong": 0.0})
    assert effects == [] and 0 < probability < 1
