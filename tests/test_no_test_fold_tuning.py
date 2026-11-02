"""Strict test fold separation: no hyperparameter tuning or search on test data.

Why this matters: if a hyperparameter (CSP components, C, shrinkage, ...) is chosen by
looking at held-out scores, the held-out score is no longer an estimate of generalisation
- it is the maximum over the candidates tried, which is optimistic, and with 9 subjects
the optimism is large. The defence is structural rather than a promise: configs are frozen
dataclasses of single values, `evaluate()` accepts nothing else, and the engine refuses
any estimator that selects hyperparameters by cross-validation. These tests fail if someone
loosens any of the three.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import pytest
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from synthetic import make_synthetic

from eeg import cv
from eeg.csp_lda import CspLdaConfig
from eeg.evaluate import default_configs, evaluate, run_splits, validate_config
from eeg.riemann import RiemannConfig


@pytest.fixture(scope="module")
def data():
    return make_synthetic()


@pytest.mark.parametrize("config", list(default_configs().values()), ids=lambda c: c.name)
def test_configs_are_frozen(config):
    assert config.__dataclass_params__.frozen
    field = dataclasses.fields(config)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, field, "changed")


def test_fixed_hyperparameters_match_the_spec():
    csp = CspLdaConfig()
    assert (csp.n_components, csp.reg, csp.log, csp.norm_trace) == (4, None, True, False)
    assert (csp.lda_solver, csp.shrinkage) == ("svd", None)
    rie = RiemannConfig()
    assert (rie.cov_estimator, rie.metric, rie.C, rie.solver, rie.max_iter, rie.seed) == (
        "oas", "riemann", 1.0, "lbfgs", 1000, 42,
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"n_components": 4},  # a dict is not a config
        [CspLdaConfig(), CspLdaConfig(n_components=6)],  # a list of candidates
        CspLdaConfig,  # the class, not an instance
        CspLdaConfig(n_components=[2, 4, 6]),  # a search space inside a config
        RiemannConfig(C=(0.1, 1.0, 10.0)),
        GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}),
    ],
    ids=["dict", "list-of-configs", "class", "csp-grid", "riemann-grid", "gridsearchcv"],
)
def test_evaluate_rejects_anything_but_one_fixed_config(data, bad):
    X, y, meta = data
    with pytest.raises(TypeError):
        evaluate(bad, X, y, meta, cv.run_grouped_splits(meta))
    with pytest.raises(TypeError):
        validate_config(bad)


def test_non_frozen_config_is_rejected(data):
    X, y, meta = data

    @dataclasses.dataclass  # not frozen
    class Loose:
        name = "loose"
        C: float = 1.0

    with pytest.raises(TypeError, match="frozen"):
        validate_config(Loose())


@pytest.mark.parametrize(
    "build",
    [
        lambda: GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}),
        lambda: Pipeline([("lr", LogisticRegressionCV())]),
    ],
    ids=["gridsearch", "cv-estimator-inside-pipeline"],
)
def test_engine_refuses_estimators_that_select_hyperparameters(data, build):
    X, y, meta = data
    with pytest.raises(TypeError, match="cross-validation"):
        run_splits(build, X, y, meta, cv.run_grouped_splits(meta))


def test_valid_config_still_runs(data):
    X, y, meta = data
    score = evaluate(default_configs()["csp_lda"], X, y, meta, cv.cross_session_splits(meta))
    assert score.n_trials == 72
