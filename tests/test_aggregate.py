"""Aggregation: correct statistics on synthetic score tables, and a hard failure on bad ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import make_subject

from eeg import evaluate as ev
from eeg.evaluate import AGG_COLUMNS, SCORE_COLUMNS, SchemaError, aggregate, sign_flip_pvalue, validate_scores

PIPELINES = ["csp_lda", "riemann_ts_lr"]


def scores(acc_by_pipeline, subjects):
    rows = [
        {"subject": s, "pipeline": p, "accuracy": a, "balanced_accuracy": a, "n_trials": 576, "chance_level": 0.25}
        for p, accs in acc_by_pipeline.items()
        for s, a in zip(subjects, accs)
    ]
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def run(df, tmp_path, subjects, pipelines=PIPELINES):
    csv = tmp_path / "per_subject_scores.csv"
    df.to_csv(csv, index=False)
    return aggregate(csv, tmp_path / "aggregate.csv", subjects, pipelines)


def test_sign_flip_pvalue_hand_computed():
    assert sign_flip_pvalue([1, 2, 3]) == pytest.approx(1 / 8)  # only +++ reaches the observed sum
    assert sign_flip_pvalue([1, 1, -1]) == pytest.approx(4 / 8)  # sums 3, 1, 1, 1 are >= 1
    assert sign_flip_pvalue([0.1] * 9) == pytest.approx(1 / 512)  # the smallest attainable p with n=9
    assert sign_flip_pvalue([-1, -2, -3]) == pytest.approx(1.0)


def test_kappa_formula_equals_cohen_kappa_when_true_classes_are_balanced():
    """Whatever the predictions look like, (acc - chance) / (1 - chance) is Cohen's kappa."""
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    rng = np.random.default_rng(0)
    y_true = np.repeat(["a", "b", "c", "d"], 144)
    for skew in ([0.25] * 4, [0.7, 0.1, 0.1, 0.1], [0.0, 0.0, 0.5, 0.5]):  # even and very uneven predictions
        y_pred = np.where(rng.random(len(y_true)) < 0.4, y_true, rng.choice(list("abcd"), len(y_true), p=skew))
        assert ev.kappa_from_accuracy(accuracy_score(y_true, y_pred), 0.25) == pytest.approx(
            cohen_kappa_score(y_true, y_pred)
        )


def test_mean_std_ddof1_and_pvalue(tmp_path):
    subjects = list(range(1, 10))
    a = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    b = [0.28, 0.33, 0.29, 0.31, 0.36, 0.30, 0.27, 0.34, 0.32]
    out = run(scores({"csp_lda": a, "riemann_ts_lr": b}, subjects), tmp_path, subjects)

    assert list(out.columns) == AGG_COLUMNS
    first = out.set_index("pipeline").loc["csp_lda"]
    assert first["mean_accuracy"] == pytest.approx(np.mean(a))
    assert first["std_accuracy"] == pytest.approx(np.std(a, ddof=1))
    assert first["std_accuracy"] != pytest.approx(np.std(a, ddof=0))
    kappa = (np.array(a) - 0.25) / 0.75
    assert first["mean_kappa"] == pytest.approx(kappa.mean())
    assert first["std_kappa"] == pytest.approx(kappa.std(ddof=1))
    assert first["n_subjects"] == 9 and first["chance_level"] == 0.25
    assert first["p_value_vs_chance"] == pytest.approx(1 / 512)  # all nine above chance
    assert first["test_used"] == ev.TEST_USED
    assert pd.read_csv(tmp_path / "aggregate.csv").columns.tolist() == AGG_COLUMNS


def test_mixed_subjects_pvalue(tmp_path):
    out = run(scores({"p": [0.35, 0.35, 0.15]}, [1, 2, 3]), tmp_path, [1, 2, 3], ["p"])
    assert out["p_value_vs_chance"].iloc[0] == pytest.approx(4 / 8)  # deviations +0.1 +0.1 -0.1


def make_good():
    subjects = list(range(1, 10))
    return subjects, scores({p: [0.3] * 9 for p in PIPELINES}, subjects)


def test_missing_row_is_a_hard_failure(tmp_path):
    subjects, df = make_good()
    with pytest.raises(SchemaError, match="missing"):
        run(df.iloc[1:], tmp_path, subjects)  # subject 1 / csp_lda dropped


@pytest.mark.parametrize(
    "damage, match",
    [
        (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
        (lambda d: d[SCORE_COLUMNS[::-1]], "columns"),
        (lambda d: d.assign(accuracy=d["accuracy"].where(d.index != 3)), "empty"),
        (lambda d: d.assign(accuracy=1.2), r"\[0, 1\]"),
        (lambda d: d.assign(n_trials=0), "positive"),
        (lambda d: d.assign(subject=d["subject"] + 100), "missing"),
    ],
    ids=["duplicate", "column-order", "nan", "accuracy>1", "no-trials", "wrong-subjects"],
)
def test_malformed_tables_are_hard_failures(tmp_path, damage, match):
    subjects, df = make_good()
    with pytest.raises(SchemaError, match=match):
        run(damage(df).reset_index(drop=True), tmp_path, subjects)


def test_unexpected_extra_subject_is_a_hard_failure(tmp_path):
    subjects, df = make_good()
    with pytest.raises(SchemaError, match="unexpected"):
        run(df, tmp_path, subjects[:-1])


def test_run_all_writes_exact_schemas_for_both_protocols(tmp_path):
    result = ev.run_all([1, 2], tmp_path, loader=make_subject)
    for target in (tmp_path, tmp_path / "cross_session"):
        assert list(pd.read_csv(target / "per_subject_scores.csv").columns) == SCORE_COLUMNS
        assert list(pd.read_csv(target / "aggregate.csv").columns) == AGG_COLUMNS
    assert (tmp_path / "hardware.json").exists()
    for png in ("accuracy_per_subject.png", "confusion.png"):
        assert (tmp_path / png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "excluded_subjects.txt").read_text() == ""
    assert not (tmp_path / "pipeline_failures.txt").exists()
    assert result["aggregate"][ev.HEADLINE]["n_subjects"].tolist() == [2, 2]


def test_failed_subject_stays_in_the_denominator_and_is_logged(tmp_path):
    def flaky(subject):
        if subject == 2:
            raise RuntimeError("corrupt file")
        return make_subject(subject)

    with pytest.raises(SchemaError, match="missing"):
        ev.run_all([1, 2], tmp_path, loader=flaky)
    assert "2\trun\tRuntimeError: corrupt file" in (tmp_path / "excluded_subjects.txt").read_text()
