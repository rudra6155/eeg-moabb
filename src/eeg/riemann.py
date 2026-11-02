"""Pipeline 2: Riemannian tangent space + logistic regression. Hyperparameters fixed (SPEC.md section 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RiemannConfig:
    name: ClassVar[str] = "riemann_ts_lr"
    cov_estimator: str = "oas"
    metric: str = "riemann"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 1000
    seed: int = 42


def build(config: RiemannConfig) -> Pipeline:
    return Pipeline(
        [
            ("cov", Covariances(estimator=config.cov_estimator)),
            # tsupdate stays False: the reference covariance is fitted on the training fold only
            ("ts", TangentSpace(metric=config.metric)),
            (
                "lr",
                LogisticRegression(
                    C=config.C, solver=config.solver, max_iter=config.max_iter, random_state=config.seed
                ),
            ),
        ]
    )
