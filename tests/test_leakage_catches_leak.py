"""The credibility anchor: the audit must fire on real leaks and stay quiet on the real pipelines.

Every test builds a deliberately leaky evaluation and asserts `LeakageError`, next to a
clean evaluation of the same data that must pass. Without the clean half, an audit that
always raises would pass this file. The data is synthetic, so none of this needs a download.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from synthetic import make_synthetic

from eeg import cv
from eeg.evaluate import build_pipeline, default_configs, run_splits
from eeg.leakage import FitWatcher, LeakageError, audit_split, audit_splits


@pytest.fixture(scope="module")
def data():
    return make_synthetic()


def _flat(A):
    return A.reshape(len(A), -1)


class _LeakyScalerPipeline:
    """Fits its scaler on ALL trials (train + test) before fitting the classifier."""

    def __init__(self, X_all):
        self.X_all = X_all

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(_flat(self.X_all))  # LEAK: test-fold statistics
        self.clf_ = LinearDiscriminantAnalysis().fit(self.scaler_.transform(_flat(X)), y)
        return self

    def predict(self, X):
        return self.clf_.predict(self.scaler_.transform(_flat(X)))


class _SneakyCspPipeline:
    """Fits CSP on exactly as many trials as the training fold has, but picks the first ones.

    The row count is right, so only the content check can see that test trials are inside.
    """

    def __init__(self, X_all, y_all):
        self.X_all, self.y_all = X_all, y_all

    def fit(self, X, y):
        n = len(X)
        self.csp_ = CSP(n_components=4).fit(self.X_all[:n], self.y_all[:n])  # LEAK
        self.clf_ = LinearDiscriminantAnalysis().fit(self.csp_.transform(X), y)
        return self

    def predict(self, X):
        return self.clf_.predict(self.csp_.transform(X))


# ---- the real thing passes -------------------------------------------------------------


@pytest.mark.parametrize("protocol", [cv.RUN_GROUPED, cv.CROSS_SESSION])
def test_real_splits_pass_the_audit(data, protocol):
    X, y, meta = data
    audit_splits(cv.make_splits(protocol, meta), meta)


@pytest.mark.parametrize("name", list(default_configs()))
def test_real_pipelines_pass_end_to_end(data, name):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    y_true, y_pred = run_splits(partial(build_pipeline, default_configs()[name]), X, y, meta, splits)
    assert len(y_true) == len(y_pred) == len(y)


# ---- split-level leaks (checks a and b) ------------------------------------------------


def test_random_trial_split_is_caught(data):
    """Random-trial k-fold puts trials of one run on both sides: the classic leak."""
    X, y, meta = data
    folds = StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
    random_splits = [cv.Split("random_trial", 1, i, tr, te) for i, (tr, te) in enumerate(folds)]
    with pytest.raises(LeakageError, match=r"\(b\)"):
        audit_splits(random_splits, meta)


def test_run_leaking_into_train_is_caught(data):
    """Move one trial of the held-out run into training: the run now sits on both sides."""
    X, y, meta = data
    good = cv.run_grouped_splits(meta)[0]
    moved = good.test_idx[0]
    bad = cv.Split(
        good.protocol, good.subject, 0,
        np.append(good.train_idx, moved), good.test_idx[1:],
    )
    audit_split(good, meta)
    with pytest.raises(LeakageError, match=r"\(b\)"):
        audit_split(bad, meta)


def test_shared_index_is_caught(data):
    X, y, meta = data
    good = cv.run_grouped_splits(meta)[0]
    bad = cv.Split(good.protocol, good.subject, 0, np.append(good.train_idx, good.test_idx[0]), good.test_idx)
    with pytest.raises(LeakageError, match=r"\(a\)"):
        audit_split(bad, meta)


def test_dropped_fold_is_caught(data):
    X, y, meta = data
    with pytest.raises(LeakageError, match="exactly once"):
        audit_splits(cv.run_grouped_splits(meta)[:-1], meta)


def test_mixed_subjects_are_caught(data):
    X, y, meta = data
    mixed = meta.copy()
    mixed.loc[0, "subject"] = 2
    with pytest.raises(LeakageError, match="mixes subjects"):
        audit_split(cv.run_grouped_splits(meta)[0], mixed)


# ---- fit-level leaks (check c) ---------------------------------------------------------


def test_scaler_fitted_on_all_data_is_caught(data):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    with pytest.raises(LeakageError, match=r"\(c\)"):
        run_splits(lambda: _LeakyScalerPipeline(X), X, y, meta, splits)


def test_csp_fitted_on_test_trials_with_matching_row_count_is_caught(data):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    with pytest.raises(LeakageError, match="test-fold trials"):
        run_splits(lambda: _SneakyCspPipeline(X, y), X, y, meta, splits)


def test_watcher_is_transparent_and_restores_the_classes(data):
    X, y, meta = data
    splits = cv.run_grouped_splits(meta)
    config = default_configs()["csp_lda"]
    before = StandardScaler.__dict__["fit"], CSP.__dict__["fit"]
    plain = build_pipeline(config).fit(X[splits[0].train_idx], y[splits[0].train_idx])
    with FitWatcher():
        watched = build_pipeline(config).fit(X[splits[0].train_idx], y[splits[0].train_idx])
    assert (StandardScaler.__dict__["fit"], CSP.__dict__["fit"]) == before
    test = X[splits[0].test_idx]
    assert np.array_equal(plain.predict(test), watched.predict(test))


def test_check_refuses_to_pass_when_nothing_was_fitted(data):
    X, y, meta = data
    with FitWatcher() as watcher:
        with pytest.raises(LeakageError, match="no fit was observed"):
            watcher.check(cv.run_grouped_splits(meta)[0], X)
