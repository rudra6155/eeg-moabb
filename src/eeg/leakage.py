"""Leakage audit.

Two layers, both of which raise `LeakageError` (an AssertionError that survives `python -O`):

1. Split audit, on indices only: `audit_split`, `audit_splits`.
   (a) no index on both sides; (b) no recording run (session + run) on both sides; plus
   subject purity, valid unique indices, protocol shape, and that run-grouped folds test
   every trial exactly once.

2. Fit audit: `FitWatcher`, for check (c). While it is active, `fit` / `fit_transform` of
   the watched preprocessing classes (StandardScaler, CSP, Covariances, TangentSpace) and of
   the classifiers is wrapped at class level, so a fit is seen no matter who calls it,
   including a pipeline that fits something on the side. After each fold, `check` requires:
     - every fit saw exactly as many rows as the training fold, and
     - every fit that was handed raw trials saw only training trials and no test trial
       (rows are compared by content hash, not by position).
   Limits: only the watched classes are observed, a fit on the right number of rows with
   the wrong content is only caught when the array still has the raw trial shape, and the
   watcher is not thread-safe (the evaluation is single-process).
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass

import numpy as np

from eeg.cv import CROSS_SESSION, RUN_GROUPED


class LeakageError(AssertionError):
    pass


def _check_indices(name: str, idx, n: int) -> np.ndarray:
    idx = np.asarray(idx)
    if idx.ndim != 1 or idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
        raise LeakageError(f"{name} must be a non-empty 1-D integer array")
    if idx.min() < 0 or idx.max() >= n:
        raise LeakageError(f"{name} holds indices outside [0, {n})")
    if np.unique(idx).size != idx.size:
        raise LeakageError(f"{name} holds duplicate indices")
    return idx


def audit_split(split, meta) -> None:
    n = len(meta)
    train = _check_indices("train_idx", split.train_idx, n)
    test = _check_indices("test_idx", split.test_idx, n)

    subjects = set(np.asarray(meta["subject"])[np.concatenate([train, test])])
    if subjects != {split.subject}:
        raise LeakageError(f"split mixes subjects: {sorted(subjects)} (expected only {split.subject})")

    shared = np.intersect1d(train, test)  # (a)
    if shared.size:
        raise LeakageError(f"(a) {shared.size} trial indices are in both train and test, e.g. {shared[:5]}")

    session = np.asarray(meta["session"]).astype(str)
    run = np.asarray(meta["run"]).astype(str)
    block = np.char.add(np.char.add(session, "/"), run)  # run ids repeat across sessions
    shared_blocks = sorted(set(block[train]) & set(block[test]))  # (b)
    if shared_blocks:
        raise LeakageError(f"(b) recording runs appear on both sides of the split: {shared_blocks}")

    if split.protocol == CROSS_SESSION and set(session[train]) & set(session[test]):
        raise LeakageError("cross_session split shares a session between train and test")
    if split.protocol == RUN_GROUPED and len(set(session[test])) != 1:
        raise LeakageError("run_grouped test fold spans several sessions")


def audit_splits(splits, meta) -> None:
    if not splits:
        raise LeakageError("no splits to audit")
    for split in splits:
        audit_split(split, meta)
    if all(s.protocol == RUN_GROUPED for s in splits):
        tested = np.sort(np.concatenate([s.test_idx for s in splits]))
        if not np.array_equal(tested, np.arange(len(meta))):
            raise LeakageError("run_grouped folds do not test every trial exactly once")


def digest_rows(array) -> list[bytes]:
    """Content hash of every row, so a trial is recognised wherever it ends up."""
    return [
        hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
        for row in np.asarray(array)
    ]


def row_digests(array) -> set[bytes]:
    return set(digest_rows(array))


@dataclass(frozen=True)
class FitRecord:
    label: str
    array: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.array)


def _watched_classes() -> tuple[type, ...]:
    from mne.decoding import CSP
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler, CSP, Covariances, TangentSpace, LinearDiscriminantAnalysis, LogisticRegression)


class FitWatcher:
    """Context manager that records the data handed to every watched fit."""

    METHODS = ("fit", "fit_transform")

    def __init__(self, classes=None):
        self.classes = tuple(classes) if classes is not None else _watched_classes()
        self.records: list[FitRecord] = []
        self._restore: list[tuple[type, str, object]] = []
        self._x_ref = None  # the X whose per-trial hashes are cached, hashed once per evaluation
        self._x_digests: list[bytes] = []

    def __enter__(self):
        for cls in self.classes:
            for name in self.METHODS:
                original = getattr(cls, name, None)
                if original is None:
                    continue
                own = cls.__dict__.get(name)  # None when the method is inherited
                setattr(cls, name, self._wrap(cls, name, original))
                self._restore.append((cls, name, own))
        return self

    def __exit__(self, *exc):
        for cls, name, own in reversed(self._restore):
            if own is None:
                delattr(cls, name)
            else:
                setattr(cls, name, own)
        self._restore.clear()
        self._x_ref, self._x_digests = None, []

    def _wrap(self, cls, name, original):
        watcher = self

        @functools.wraps(original)
        def wrapper(self, X, *args, **kwargs):
            watcher.records.append(FitRecord(f"{cls.__name__}.{name}", np.asarray(X)))
            return original(self, X, *args, **kwargs)

        return wrapper

    def reset(self) -> None:
        self.records.clear()

    def check(self, split, X) -> None:
        """Audit the fits recorded since the last reset against this split (check (c))."""
        if not self.records:
            raise LeakageError("(c) no fit was observed, so nothing was audited")
        n_train, raw_shape = len(split.train_idx), np.shape(X)[1:]
        train_digests = test_digests = None
        by_array: dict[tuple, set[bytes]] = {}  # fit_transform and fit receive the same array
        saw_raw = False
        for rec in self.records:
            if rec.rows != n_train:
                raise LeakageError(
                    f"(c) {rec.label} was fitted on {rec.rows} rows, but the training fold has {n_train}"
                )
            if rec.array.shape[1:] != raw_shape:
                continue  # derived features: row count (checked above) is all that can be said
            saw_raw = True
            if train_digests is None:
                if self._x_ref is not X:
                    self._x_ref, self._x_digests = X, digest_rows(X)
                train_digests = {self._x_digests[i] for i in split.train_idx}
                test_digests = {self._x_digests[i] for i in split.test_idx}
            key = (rec.array.__array_interface__["data"][0], rec.array.shape, rec.array.strides)
            if key not in by_array:
                by_array[key] = row_digests(rec.array)
            fitted = by_array[key]
            seen_test = fitted & test_digests
            if seen_test:
                raise LeakageError(f"(c) {rec.label} was fitted on {len(seen_test)} test-fold trials")
            if not fitted <= train_digests:
                raise LeakageError(f"(c) {rec.label} was fitted on trials that are not in the training fold")
        if not saw_raw:
            raise LeakageError("(c) no fit received raw trials, so the content check did not run")
