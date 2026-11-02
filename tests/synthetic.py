"""Small synthetic stand-in for one BNCI2014_001 subject: same structure, no download.

2 sessions x 6 runs x (4 classes x trials_per_class_per_run) trials. Class k raises the
variance of channel k, which is exactly the signal CSP and covariance methods can use, so
the real pipelines learn it and the shapes and the audit are exercised on the real code paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg.data import SubjectData

CLASSES = np.array(["feet", "left_hand", "right_hand", "tongue"])


def make_synthetic(trials_per_class_per_run=3, n_channels=6, n_samples=64, boost=3.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y, rows = [], [], []
    for session in ("0train", "1test"):
        for run in range(6):
            labels = np.repeat(CLASSES, trials_per_class_per_run)
            rng.shuffle(labels)
            for label in labels:
                x = rng.standard_normal((n_channels, n_samples))
                x[int(np.flatnonzero(CLASSES == label)[0])] *= boost
                X.append(x)
                y.append(label)
                rows.append((1, session, str(run)))
    meta = pd.DataFrame(rows, columns=["subject", "session", "run"])
    return np.stack(X), np.array(y), meta


def make_subject(subject: int, **kwargs) -> SubjectData:
    X, y, meta = make_synthetic(seed=subject, **kwargs)
    meta = meta.assign(subject=subject)
    return SubjectData(subject, X, y, meta, 250.0)
