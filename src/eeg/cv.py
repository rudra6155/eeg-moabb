"""Cross-validation splits for one subject.

Every split is returned as an explicit `Split` (train and test trial indices into that
subject's X), so leakage.py can audit exactly what the evaluation will use.

Two protocols, both strictly per subject (subjects are never pooled):

* run_grouped (headline): within each session, leave one recording run out
  (sklearn LeaveOneGroupOut over run ids). 2 sessions x 6 runs = 12 folds, and every trial
  is tested exactly once.
* cross_session (secondary, the BCI Competition IV 2a protocol): train on session 0train,
  test on session 1test. One fold.

Neither is MOABB's WithinSessionEvaluation / CrossSessionEvaluation class: those return
aggregated scores, but the audit and the balanced-accuracy scoring need the indices and
the out-of-fold predictions. See SPEC.md section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

RUN_GROUPED = "run_grouped"
CROSS_SESSION = "cross_session"
TRAIN_SESSION = "0train"
TEST_SESSION = "1test"


@dataclass(frozen=True)
class Split:
    protocol: str
    subject: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str = ""

    def __post_init__(self):
        for name in ("train_idx", "test_idx"):
            arr = np.array(getattr(self, name), dtype=np.int64)  # private, read-only copy
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)


def _subject_of(meta) -> int:
    subjects = set(meta["subject"])
    if len(subjects) != 1:
        raise ValueError(f"splits are per subject, but the metadata holds subjects {sorted(subjects)}")
    return int(subjects.pop())


def run_grouped_splits(meta) -> list[Split]:
    """Leave-one-run-out within each session.

    Run ids ('0'..'5') repeat in every session, so grouping by run id alone would pool two
    different physical runs. The groups are therefore formed inside each session.
    """
    subject = _subject_of(meta)
    session = np.asarray(meta["session"])
    run = np.asarray(meta["run"])
    splits: list[Split] = []
    for s in sorted(set(session)):
        in_session = np.flatnonzero(session == s)
        for train, test in LeaveOneGroupOut().split(in_session, groups=run[in_session]):
            held_out = run[in_session[test]][0]
            splits.append(
                Split(
                    RUN_GROUPED,
                    subject,
                    len(splits),
                    in_session[train],
                    in_session[test],
                    f"session={s}, held-out run={held_out}",
                )
            )
    return splits


def cross_session_splits(meta) -> list[Split]:
    """Train on session 0train, test on session 1test (one direction only)."""
    subject = _subject_of(meta)
    session = np.asarray(meta["session"])
    train, test = np.flatnonzero(session == TRAIN_SESSION), np.flatnonzero(session == TEST_SESSION)
    if train.size == 0 or test.size == 0:
        raise ValueError(f"need sessions {TRAIN_SESSION!r} and {TEST_SESSION!r}, found {sorted(set(session))}")
    return [Split(CROSS_SESSION, subject, 0, train, test, f"train={TRAIN_SESSION}, test={TEST_SESSION}")]


SPLITTERS = {RUN_GROUPED: run_grouped_splits, CROSS_SESSION: cross_session_splits}


def make_splits(protocol: str, meta) -> list[Split]:
    return SPLITTERS[protocol](meta)
