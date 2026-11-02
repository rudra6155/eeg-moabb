"""Label-shuffle negative control (CI gate).

With labels permuted within a subject there is nothing to decode, so a leak-free pipeline
must score at chance. The real-data test runs the full pipeline end to end (run-grouped CV,
audit on) on 2 subjects, and also scores the TRUE labels: a pipeline that is simply broken
would also sit at chance on shuffled labels, so the unshuffled run must be clearly above it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeg import cv
from eeg.controls import CONTROL_COLUMNS, judge, run_controls, shuffle_labels
from eeg.csp_lda import CspLdaConfig
from eeg.evaluate import evaluate
from synthetic import make_subject

CHANCE = 0.25


def test_shuffle_is_deterministic_balanced_per_session_and_actually_shuffles():
    session = np.repeat(["0train", "1test"], 200)
    y = np.tile(np.repeat(["a", "b", "c", "d"], 50), 2)
    first = shuffle_labels(y, session, subject=3, seed=42)
    assert np.array_equal(first, shuffle_labels(y, session, subject=3, seed=42))
    for s in ("0train", "1test"):  # class counts unchanged inside each session
        assert sorted(first[session == s]) == sorted(y[session == s])
    assert not np.array_equal(first, y)
    assert not np.array_equal(first, shuffle_labels(y, session, subject=4, seed=42))
    assert not np.array_equal(first, shuffle_labels(y, session, subject=3, seed=43))


def test_judge_flags_above_chance_and_passes_chance():
    def table(acc):
        return pd.DataFrame(
            {"control": "label_shuffle", "pipeline": "csp_lda", "subject": range(1, 10),
             "accuracy": acc, "n_trials": 576, "chance_level": CHANCE}
        )

    assert judge(table(0.25)) == []
    assert judge(table(0.27)) == []
    assert judge(table(0.30))  # mean 0.05 above chance: stop
    one_bad = [0.25] * 8 + [0.40]
    assert any("subject 9" in p for p in judge(table(one_bad), aggregate_tolerance=0.10))


def test_controls_table_has_the_exact_columns_on_synthetic_data():
    df = run_controls([1, 2], loader=make_subject)
    assert list(df.columns) == CONTROL_COLUMNS
    assert set(df["control"]) == {"label_shuffle", "label_shuffle_cross_session"}
    assert len(df) == 2 * 2 * 2  # subjects x protocols x pipelines


def test_label_shuffle_is_at_chance_and_true_labels_are_not(real_subject):
    """Shuffled labels: headline protocol (12 audited folds). True labels: one cross-session
    fold, which is enough to show the pipeline learns and keeps `make test` under its budget."""
    config = CspLdaConfig()
    shuffled, true = [], []
    for subject in (1, 2):
        data = real_subject(subject)
        y_shuffled = shuffle_labels(data.y, data.meta["session"], subject)
        grouped = cv.run_grouped_splits(data.meta)
        shuffled.append(evaluate(config, data.X, y_shuffled, data.meta, grouped).accuracy)
        cross = cv.cross_session_splits(data.meta)
        true.append(evaluate(config, data.X, data.y, data.meta, cross).accuracy)

    assert all(abs(a - CHANCE) < 0.10 for a in shuffled), f"shuffled accuracies {shuffled}"
    assert np.mean(true) > CHANCE + 0.10, f"true-label accuracies {true}: the pipeline is not learning"
