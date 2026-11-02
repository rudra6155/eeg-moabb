"""Determinism (test matrix 4).

The real pipelines have no stochastic component (closed-form CSP/LDA, deterministic lbfgs,
fixed leave-one-run-out folds), so the seed cannot change their accuracy and the right
assertion is that it does not. The only seeded randomness is the label-shuffle control,
which is where 'different seed => different but near-identical (at chance)' is tested.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_synthetic

from eeg import cv
from eeg.controls import shuffle_labels
from eeg.evaluate import default_configs, evaluate


@pytest.fixture(scope="module")
def data():
    return make_synthetic()


@pytest.mark.parametrize("name", list(default_configs()))
def test_same_seed_gives_identical_predictions(data, name):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    a = evaluate(default_configs()[name], X, y, meta, splits, seed=7)
    b = evaluate(default_configs()[name], X, y, meta, splits, seed=7)
    assert a.accuracy == b.accuracy
    assert np.array_equal(a.y_pred, b.y_pred)


@pytest.mark.parametrize("name", list(default_configs()))
def test_pipelines_do_not_depend_on_the_seed(data, name):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    a = evaluate(default_configs()[name], X, y, meta, splits, seed=1)
    b = evaluate(default_configs()[name], X, y, meta, splits, seed=2)
    assert a.accuracy == b.accuracy


def test_shuffle_seed_changes_the_labels_but_both_runs_stay_at_chance(data):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    config = default_configs()["csp_lda"]
    y1 = shuffle_labels(y, meta["session"], 1, seed=1)
    y2 = shuffle_labels(y, meta["session"], 1, seed=2)
    assert not np.array_equal(y1, y2)
    for shuffled in (y1, y2):
        assert abs(evaluate(config, X, shuffled, meta, splits).accuracy - 0.25) < 0.15
