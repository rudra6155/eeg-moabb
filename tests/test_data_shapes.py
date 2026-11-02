"""Loaded data must match results/dataset_manifest.json; skipped when the dataset is absent."""

from __future__ import annotations

import numpy as np
import pytest

from eeg import cv
from eeg.data import load_manifest
from eeg.evaluate import chance_level

SUBJECTS = [1, 2]


@pytest.mark.parametrize("subject", SUBJECTS)
def test_subject_matches_manifest(real_subject, subject):
    data = real_subject(subject)
    manifest = load_manifest()
    entry = manifest["subjects"][str(subject)]

    assert list(data.X.shape) == entry["X_shape"]
    assert data.y.shape == (entry["n_trials"],)
    assert len(data.meta) == entry["n_trials"]
    assert data.sfreq == entry["sampling_rate_hz"] == manifest["sampling_rate_hz"]
    assert sorted(set(data.y)) == manifest["classes"]
    assert len(set(data.y)) == manifest["n_classes"]
    for session, info in entry["sessions"].items():
        in_session = data.meta["session"] == session
        assert in_session.sum() == info["n_trials"]
        for run, n in info["runs"].items():
            assert (in_session & (data.meta["run"] == run)).sum() == n


@pytest.mark.parametrize("subject", SUBJECTS)
def test_chance_level_is_one_over_n_classes_and_matches_class_balance(real_subject, subject):
    data = real_subject(subject)
    manifest = load_manifest()
    chance = chance_level(data.y)  # raises if the classes are not balanced
    assert chance == pytest.approx(1 / len(set(data.y)))
    assert chance == pytest.approx(manifest["chance_level"])
    _, counts = np.unique(data.y, return_counts=True)
    assert counts.max() / counts.sum() == pytest.approx(chance)


@pytest.mark.parametrize("subject", SUBJECTS)
def test_real_splits_have_the_expected_shape(real_subject, subject):
    meta = real_subject(subject).meta
    grouped = cv.run_grouped_splits(meta)
    assert len(grouped) == 12
    assert {len(s.test_idx) for s in grouped} == {48}
    assert {len(s.train_idx) for s in grouped} == {240}
    (cross,) = cv.cross_session_splits(meta)
    assert (len(cross.train_idx), len(cross.test_idx)) == (288, 288)


def test_chance_level_refuses_imbalanced_classes():
    with pytest.raises(ValueError, match="not balanced"):
        chance_level(np.array(["a"] * 30 + ["b"] * 10))
