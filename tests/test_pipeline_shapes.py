"""Both pipelines produce the shapes the SPEC promises and learn a planted signal."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import make_synthetic

from eeg import cv
from eeg.evaluate import SCORE_COLUMNS, build_pipeline, default_configs, evaluate


@pytest.fixture(scope="module")
def data():
    return make_synthetic()


def test_csp_lda_feature_and_output_shapes(data):
    X, y, meta = data
    pipe = build_pipeline(default_configs()["csp_lda"]).fit(X, y)
    assert pipe[0].transform(X).shape == (len(X), 4)  # CSP(4 components)
    assert pipe.predict(X).shape == (len(X),)
    assert set(pipe.predict(X)) <= set(y)


def test_riemann_feature_and_output_shapes(data):
    X, y, meta = data
    n_ch = X.shape[1]
    pipe = build_pipeline(default_configs()["riemann_ts_lr"]).fit(X, y)
    assert pipe[0].transform(X).shape == (len(X), n_ch, n_ch)  # covariances
    assert pipe[:2].transform(X).shape == (len(X), n_ch * (n_ch + 1) // 2)  # tangent space
    assert pipe.predict(X).shape == (len(X),)


@pytest.mark.parametrize("name", list(default_configs()))
@pytest.mark.parametrize("protocol, n_scored", [(cv.RUN_GROUPED, 144), (cv.CROSS_SESSION, 72)])
def test_evaluate_scores_the_right_trials_and_learns(data, name, protocol, n_scored):
    X, y, meta = data
    score = evaluate(default_configs()[name], X, y, meta, cv.make_splits(protocol, meta))
    assert score.n_trials == n_scored == len(score.y_true) == len(score.y_pred)
    assert list(score.row()) == SCORE_COLUMNS
    assert score.chance_level == 0.25
    assert score.accuracy > 0.5  # planted variance signal: a working pipeline finds it
    if protocol == cv.RUN_GROUPED:  # every trial tested exactly once
        assert sorted(score.y_true) == sorted(y)
