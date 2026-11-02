"""Pipeline 1: CSP (4 components) + LDA. Every hyperparameter is fixed here (SPEC.md section 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class CspLdaConfig:
    name: ClassVar[str] = "csp_lda"
    n_components: int = 4
    reg: float | None = None
    log: bool = True
    norm_trace: bool = False
    lda_solver: str = "svd"
    shrinkage: float | None = None


def build(config: CspLdaConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=config.n_components,
                    reg=config.reg,
                    log=config.log,
                    norm_trace=config.norm_trace,
                ),
            ),
            ("lda", LinearDiscriminantAnalysis(solver=config.lda_solver, shrinkage=config.shrinkage)),
        ]
    )
