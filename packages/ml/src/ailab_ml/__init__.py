# ==================================================================================================
# 📜 ailab-ml
# --------------------------------------------------------------------------------------------------
#  File         : packages/ml/src/ailab_ml/__init__.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Entry point of the AI Lab machine-learning core: explainable models and the validation
#    metrics used to approve them, shared by every framework (finance, health, climate ...).
#
#  🔧 WHAT IT DOES
#    Re-exports logistic regression and the validation metrics.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: LogisticRegression (IRLS), AUC, Gini, KS, Brier, calibration, PSI.
# ==================================================================================================
"""ailab-ml: explainable models and model-validation metrics (NumPy only)."""

# ========================================
# 📂 Imports
# ========================================
from ailab_ml.linear import LogisticRegression, logistic_irls, sigmoid
from ailab_ml.metrics import (
    CalibrationBin,
    brier_score,
    calibration_table,
    gini,
    ks_statistic,
    population_stability_index,
    roc_auc,
)

# ========================================
# 📂 Package Metadata
# ========================================
__version__ = "1.0.0"
__author__ = "Mathew Varghese"

# ========================================
# 📂 Public API
# ========================================
__all__ = [
    "CalibrationBin",
    "LogisticRegression",
    "brier_score",
    "calibration_table",
    "gini",
    "ks_statistic",
    "logistic_irls",
    "population_stability_index",
    "roc_auc",
    "sigmoid",
]
