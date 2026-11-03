# ==================================================================================================
# 📜 ailab-ml :: linear
# --------------------------------------------------------------------------------------------------
#  File         : packages/ml/src/ailab_ml/linear.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Regularised logistic regression: still the workhorse of credit scoring, medical risk scores
#    and any setting where a probability must be explained to a regulator, a clinician or a
#    customer. Small, exact and dependency-free (NumPy only).
#
#  🔧 WHAT IT DOES
#    - sigmoid            : numerically safe logistic function.
#    - logistic_irls      : Newton / IRLS solver on a design matrix whose first column is the
#                           intercept (left unpenalised).
#    - LogisticRegression : fit / predict_proba with optional standardisation and L2 penalty.
#
#  🧠 DESIGN NOTES
#    - Newton's method (IRLS) converges in a handful of iterations on problems of this size and
#      gives the exact penalised maximum-likelihood solution; no learning rate to tune.
#    - The intercept is never penalised, so regularisation shrinks effects, not the base rate.
#    - Standardisation is optional. Scorecards fit on Weight-of-Evidence features, which are
#      already on a common scale and whose raw coefficients become points, so they turn it off.
#    - Inputs are validated: binary targets only, both classes present, no NaN. A model fit on
#      one class would silently predict a constant.
#    - Extracted from the finance forecast model when the credit scorecard became its second
#      user (AI Lab ADR-0003).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: sigmoid, logistic_irls, LogisticRegression.
# ==================================================================================================
"""L2-regularised logistic regression (IRLS), NumPy only.

Example:
    >>> import numpy as np
    >>> x = np.array([[0.0], [1.0], [2.0], [3.0]] * 10)
    >>> y = np.array([0, 0, 1, 1] * 10)
    >>> model = LogisticRegression(l2=0.1).fit(x, y)
    >>> bool(model.predict_proba([[3.0]])[0] > 0.9)
    True
"""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


# ========================================
# 📂 Core Functions
# ========================================
def sigmoid(z: npt.ArrayLike) -> FloatArray:
    """1 / (1 + e^-z), clipped so large |z| never overflows."""
    out: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -30, 30)))
    return out


def logistic_irls(
    design: npt.ArrayLike,
    targets: npt.ArrayLike,
    *,
    l2: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> FloatArray:
    """Penalised logistic regression weights. Column 0 of ``design`` is the intercept."""
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(x @ weights)
        hessian = (x.T * (p * (1 - p))) @ x + np.diag(penalty + 1e-9)
        gradient = x.T @ (y - p) - penalty * weights
        step = np.linalg.solve(hessian, gradient)
        weights += step
        if np.max(np.abs(step)) < tol:
            break
    return weights


# ========================================
# 📂 Estimator
# ========================================
class LogisticRegression:
    """Binary logistic regression with L2 penalty and optional standardisation."""

    def __init__(self, *, l2: float = 1.0, standardize: bool = True) -> None:
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.l2 = l2
        self.standardize = standardize
        self._weights: FloatArray | None = None

    def _design(self, x: FloatArray) -> FloatArray:
        if self.standardize:
            x = (x - self._mean) / self._scale
        return np.hstack([np.ones((len(x), 1)), x])

    def fit(self, features: npt.ArrayLike, targets: npt.ArrayLike) -> LogisticRegression:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be 2-D with one row per target")
        if np.isnan(x).any() or not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("features must not contain NaN and targets must be 0 or 1")
        if len(np.unique(y)) < 2:
            raise ValueError("targets contain only one class")
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        self._weights = logistic_irls(self._design(x), y, l2=self.l2)
        return self

    def _fitted_weights(self) -> FloatArray:
        if self._weights is None:
            raise RuntimeError("call fit() first")
        return self._weights

    @property
    def intercept(self) -> float:
        return float(self._fitted_weights()[0])

    @property
    def coefficients(self) -> FloatArray:
        """Per-feature weights (on the standardised scale when standardize=True)."""
        weights: FloatArray = self._fitted_weights()[1:].copy()
        return weights

    def predict_proba(self, features: npt.ArrayLike) -> FloatArray:
        """Probability of class 1 for each row."""
        weights = self._fitted_weights()
        x = np.asarray(features, dtype=np.float64)
        return sigmoid(self._design(x.reshape(len(x), -1)) @ weights)
