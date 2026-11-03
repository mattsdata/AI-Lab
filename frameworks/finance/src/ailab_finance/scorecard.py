# ==================================================================================================
# 📜 ailab-finance :: scorecard
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/src/ailab_finance/scorecard.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A credit scorecard as banks build, validate and explain them: Weight-of-Evidence binning,
#    logistic regression, points scaled so that every 20 points doubles the odds, reason codes
#    for each decision, a validation pack, and a fairness check across groups.
#
#  🔧 WHAT IT DOES
#    - WoeBinning       : quantile bins per feature, missing values in their own bin, WoE, IV,
#                         and a monotonicity check.
#    - Scorecard        : selects features by Information Value, fits ailab-ml logistic regression
#                         on WoE, converts to points (base score / base odds / PDO), predicts PD
#                         and score, explains decisions with reason codes, prints the points table.
#    - validate         : AUC, Gini, KS, Brier, calibration by decile and PSI vs the build sample.
#    - group_outcomes   : approval and default rates per group at a cut-off, and the adverse
#                         impact ratio (lowest approval rate / highest).
#
#  🧠 DESIGN NOTES
#    - WoE = ln(share of non-defaulters / share of defaulters) per bin, with +0.5 smoothing so an
#      all-good or all-bad bin stays finite. Higher WoE means lower risk.
#    - Points are additive per characteristic, so a customer or a regulator can see exactly
#      which answer cost how many points. Reason codes are the characteristics with the largest
#      shortfall from their best possible points.
#    - Features below `min_iv` (default 0.02, the conventional "not predictive" line) are
#      dropped and listed, never silently. IV is biased upward in small samples (pure noise can
#      reach 0.09 on 500 rows, see tests), so judge IV on thousands of rows or a hold-out.
#    - Protected characteristics are never model inputs: `fit` has no slot for them. They are
#      used only afterwards, in group_outcomes, to test the decisions for disparate impact.
#    - The "four-fifths" ratio is a screening heuristic, not a legal test; fair-lending law
#      differs by jurisdiction (e.g. FCA Consumer Duty and the Equality Act in the UK).
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: WoE binning, points scorecard, reason codes, validation, fairness.
# ==================================================================================================
"""Credit scorecard: WoE binning, points, reason codes, validation and fairness."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ailab_ml import (
    CalibrationBin,
    LogisticRegression,
    brier_score,
    calibration_table,
    gini,
    ks_statistic,
    population_stability_index,
    roc_auc,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
MISSING = -1  # bin index used for missing values


# ========================================
# 📂 Weight-of-Evidence Binning
# ========================================
@dataclass(frozen=True)
class WoeBinning:
    """Bins for one characteristic. Bin k covers (edges[k-1], edges[k]]; NaN is MISSING."""

    name: str
    edges: tuple[float, ...]  # inner cut points, ascending
    woe: tuple[float, ...]  # one value per bin
    counts: tuple[int, ...]
    event_rates: tuple[float, ...]
    missing_woe: float  # 0.0 if no missing values were seen in training
    missing_count: int
    information_value: float

    @classmethod
    def fit(
        cls,
        name: str,
        values: npt.ArrayLike,
        defaults: npt.ArrayLike,
        *,
        bins: int = 5,
        smoothing: float = 0.5,
    ) -> WoeBinning:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(defaults, dtype=np.float64)
        present = ~np.isnan(x)
        if present.sum() < bins:
            raise ValueError(f"{name}: too few non-missing values for {bins} bins")
        quantiles = np.quantile(x[present], np.linspace(0, 1, bins + 1)[1:-1])
        edges = tuple(float(e) for e in np.unique(quantiles))
        index = np.where(present, np.searchsorted(edges, x, side="left"), MISSING)

        total_bad = y.sum() + smoothing
        total_good = (1 - y).sum() + smoothing

        def woe_of(mask: npt.NDArray[np.bool_]) -> tuple[float, float]:
            bad, good = y[mask].sum() + smoothing, (1 - y[mask]).sum() + smoothing
            dist_bad, dist_good = bad / total_bad, good / total_good
            woe = math.log(dist_good / dist_bad)
            return woe, (dist_good - dist_bad) * woe

        woe, counts, rates, iv = [], [], [], 0.0
        for k in range(len(edges) + 1):
            mask = index == k
            w, contribution = woe_of(mask)
            woe.append(w)
            counts.append(int(mask.sum()))
            rates.append(float(y[mask].mean()) if mask.any() else 0.0)
            iv += contribution
        missing = index == MISSING
        missing_woe = 0.0
        if missing.any():
            missing_woe, contribution = woe_of(missing)
            iv += contribution
        return cls(
            name,
            edges,
            tuple(woe),
            tuple(counts),
            tuple(rates),
            missing_woe,
            int(missing.sum()),
            iv,
        )

    def bin_index(self, values: npt.ArrayLike) -> IntArray:
        x = np.asarray(values, dtype=np.float64)
        index: IntArray = np.where(
            np.isnan(x), MISSING, np.searchsorted(self.edges, x, side="left")
        ).astype(np.int64)
        return index

    def transform(self, values: npt.ArrayLike) -> FloatArray:
        index = self.bin_index(values)
        lookup = np.array(self.woe)
        out: FloatArray = np.where(index == MISSING, self.missing_woe, lookup[np.maximum(index, 0)])
        return out

    @property
    def is_monotonic(self) -> bool:
        """True if WoE rises or falls steadily across the ordered bins (a common review rule)."""
        steps = np.diff(self.woe)
        return bool(np.all(steps >= 0) or np.all(steps <= 0))

    def labels(self) -> list[str]:
        bounds = [-math.inf, *self.edges, math.inf]
        return [f"({bounds[k]:g}, {bounds[k + 1]:g}]" for k in range(len(bounds) - 1)]


# ========================================
# 📂 Scorecard
# ========================================
@dataclass(frozen=True)
class Explanation:
    score: float
    probability_of_default: float
    points: dict[str, float]  # points earned per characteristic
    reasons: list[str]  # characteristics that cost the most points, worst first


class Scorecard:
    """Points scorecard. Default: 600 points = odds of 50:1 (good:bad); +20 points doubles odds."""

    def __init__(
        self,
        *,
        bins: int = 5,
        l2: float = 1.0,
        min_iv: float = 0.02,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        pdo: float = 20.0,
    ) -> None:
        if pdo <= 0 or base_odds <= 0:
            raise ValueError("pdo and base_odds must be positive")
        self._bins = bins
        self._l2 = l2
        self.min_iv = min_iv
        self.factor = pdo / math.log(2)
        self.offset = base_score - self.factor * math.log(base_odds)
        self.binnings: dict[str, WoeBinning] = {}
        self.dropped: dict[str, float] = {}  # feature -> IV, below min_iv
        self._model: LogisticRegression | None = None

    def fit(self, features: Mapping[str, npt.ArrayLike], defaults: npt.ArrayLike) -> Scorecard:
        """Fit on characteristics (name -> values, NaN allowed) and default flags (1 = default)."""
        y = np.asarray(defaults, dtype=np.float64)
        candidates = {
            name: WoeBinning.fit(name, values, y, bins=self._bins)
            for name, values in features.items()
        }
        self.binnings = {n: b for n, b in candidates.items() if b.information_value >= self.min_iv}
        self.dropped = {
            n: b.information_value for n, b in candidates.items() if n not in self.binnings
        }
        if not self.binnings:
            raise ValueError("no characteristic reached the minimum information value")
        self._model = LogisticRegression(l2=self._l2, standardize=False).fit(
            self._woe_matrix(features), y
        )
        return self

    def _fitted(self) -> LogisticRegression:
        if self._model is None:
            raise RuntimeError("call fit() first")
        return self._model

    def _woe_matrix(self, features: Mapping[str, npt.ArrayLike]) -> FloatArray:
        missing = [n for n in self.binnings if n not in features]
        if missing:
            raise KeyError(f"missing characteristics: {', '.join(missing)}")
        return np.column_stack([b.transform(features[n]) for n, b in self.binnings.items()])

    @property
    def base_points(self) -> float:
        return self.offset - self.factor * self._fitted().intercept

    def points(self, features: Mapping[str, npt.ArrayLike]) -> dict[str, FloatArray]:
        """Points earned per characteristic for each row."""
        coefficients = self._fitted().coefficients
        woe = self._woe_matrix(features)
        return {
            name: -self.factor * coefficients[j] * woe[:, j] for j, name in enumerate(self.binnings)
        }

    def score(self, features: Mapping[str, npt.ArrayLike]) -> FloatArray:
        total: FloatArray = self.base_points + np.sum(list(self.points(features).values()), axis=0)
        return total

    def probability_of_default(self, features: Mapping[str, npt.ArrayLike]) -> FloatArray:
        return self._fitted().predict_proba(self._woe_matrix(features))

    def explain(self, applicant: Mapping[str, float], *, top: int = 3) -> Explanation:
        """Score, PD, points per characteristic and the main reasons points were lost."""
        row = {name: [value] for name, value in applicant.items()}
        earned = {name: float(p[0]) for name, p in self.points(row).items()}
        coefficients = self._fitted().coefficients
        best = {
            name: max(-self.factor * coefficients[j] * w for w in (*b.woe, b.missing_woe))
            for j, (name, b) in enumerate(self.binnings.items())
        }
        shortfall = sorted(earned, key=lambda n: best[n] - earned[n], reverse=True)
        reasons = [n for n in shortfall if best[n] - earned[n] > 1e-9][:top]
        return Explanation(
            score=float(self.score(row)[0]),
            probability_of_default=float(self.probability_of_default(row)[0]),
            points=earned,
            reasons=reasons,
        )

    def table(self) -> str:
        """The scorecard as a points table, one line per characteristic and bin."""
        coefficients = self._fitted().coefficients
        lines = [f"Base points: {self.base_points:.0f}"]
        for j, (name, b) in enumerate(self.binnings.items()):
            flag = "" if b.is_monotonic else "  [WoE not monotonic: review bins]"
            lines.append(f"{name}  (IV {b.information_value:.3f}){flag}")
            for label, w, n, rate in zip(b.labels(), b.woe, b.counts, b.event_rates, strict=True):
                pts = -self.factor * coefficients[j] * w
                lines.append(
                    f"  {label:<24} n={n:<6} default {rate:6.1%}  WoE {w:+.2f}  {pts:+.0f}"
                )
            if b.missing_count:
                pts = -self.factor * coefficients[j] * b.missing_woe
                lines.append(f"  {'missing':<24} n={b.missing_count:<6} {'':<16}  {pts:+.0f}")
        for name, iv in self.dropped.items():
            lines.append(f"{name}  dropped (IV {iv:.3f} < {self.min_iv})")
        return "\n".join(lines)


# ========================================
# 📂 Validation Pack
# ========================================
@dataclass(frozen=True)
class ValidationReport:
    auc: float
    gini: float
    ks: float
    brier: float
    calibration: list[CalibrationBin]
    psi: float | None  # score stability vs reference; None if no reference given

    def summary(self) -> str:
        lines = [
            f"AUC {self.auc:.3f} · Gini {self.gini:.3f} · KS {self.ks:.3f} · Brier {self.brier:.4f}"
        ]
        if self.psi is not None:
            reading = "stable" if self.psi < 0.10 else "monitor" if self.psi < 0.25 else "shifted"
            lines.append(f"PSI vs build sample: {self.psi:.3f} ({reading})")
        lines.append("Calibration (predicted vs observed default rate by decile):")
        lines += [f"  {b.predicted:6.2%}  {b.observed:6.2%}  n={b.count}" for b in self.calibration]
        return "\n".join(lines)


def validate(
    scorecard: Scorecard,
    features: Mapping[str, npt.ArrayLike],
    defaults: npt.ArrayLike,
    *,
    reference_scores: npt.ArrayLike | None = None,
) -> ValidationReport:
    """Validation pack on a hold-out sample (never the sample the scorecard was fitted on)."""
    y = np.asarray(defaults, dtype=np.float64)
    pd_ = scorecard.probability_of_default(features)
    psi = None
    if reference_scores is not None:
        psi = population_stability_index(reference_scores, scorecard.score(features))
    return ValidationReport(
        auc=roc_auc(y, pd_),
        gini=gini(y, pd_),
        ks=ks_statistic(y, pd_),
        brier=brier_score(y, pd_),
        calibration=calibration_table(y, pd_, bins=10),
        psi=psi,
    )


# ========================================
# 📂 Fairness Check
# ========================================
@dataclass(frozen=True, slots=True)
class GroupOutcome:
    count: int
    approval_rate: float
    default_rate_approved: float  # among approved applicants; NaN if none approved


def group_outcomes(
    scores: npt.ArrayLike,
    defaults: npt.ArrayLike,
    groups: Sequence[str],
    *,
    cutoff: float,
) -> tuple[dict[str, GroupOutcome], float]:
    """Approval and default rates per group at ``cutoff``, and the adverse impact ratio."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(defaults, dtype=np.float64)
    g = np.asarray(groups)
    if not len(s) == len(y) == len(g):
        raise ValueError("scores, defaults and groups must have the same length")
    outcomes = {}
    for name in sorted(set(g.tolist())):
        mask = g == name
        approved = mask & (s >= cutoff)
        outcomes[str(name)] = GroupOutcome(
            count=int(mask.sum()),
            approval_rate=float(approved.sum() / mask.sum()),
            default_rate_approved=float(y[approved].mean()) if approved.any() else math.nan,
        )
    rates = [o.approval_rate for o in outcomes.values()]
    ratio = min(rates) / max(rates) if max(rates) > 0 else math.nan
    return outcomes, ratio
