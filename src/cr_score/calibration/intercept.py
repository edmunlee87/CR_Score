"""
Intercept calibration for probability adjustments.

Adjusts model intercept to match target population bad rate.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cr_score.core.logging import get_audit_logger


class InterceptCalibrator:
    """
    Calibrate model intercept to match target bad rate.

    Useful when model trained on different population or time period.

    Example:
        >>> calibrator = InterceptCalibrator(target_bad_rate=0.05)
        >>> calibrator.fit(y_pred_proba_dev, y_true_dev)
        >>> y_pred_calibrated = calibrator.transform(y_pred_proba_oot)
    """

    def __init__(self, target_bad_rate: Optional[float] = None) -> None:
        """
        Initialize intercept calibrator.

        Args:
            target_bad_rate: Target bad rate (if None, uses observed rate)
        """
        self.target_bad_rate = target_bad_rate
        self.logger = get_audit_logger()

        self.intercept_adjustment_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        y_pred_proba: np.ndarray,
        y_true: Optional[pd.Series] = None,
    ) -> "InterceptCalibrator":
        """
        Fit intercept adjustment.

        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels (required if target_bad_rate not set)

        Returns:
            Self

        Example:
            >>> calibrator.fit(predictions_dev, y_dev)
        """
        if self.target_bad_rate is None:
            if y_true is None:
                raise ValueError("Either target_bad_rate or y_true must be provided")
            self.target_bad_rate = float(y_true.mean())

        self.logger.info(
            "Fitting intercept calibration",
            target_bad_rate=self.target_bad_rate,
        )

        # Find intercept adjustment that achieves target bad rate
        def objective(adjustment: float) -> float:
            adjusted_proba = self._apply_adjustment(y_pred_proba, adjustment)
            return abs(adjusted_proba.mean() - self.target_bad_rate)

        result = minimize_scalar(objective, bounds=(-5, 5), method="bounded")

        self.intercept_adjustment_ = result.x

        self.is_fitted_ = True

        self.logger.info(
            "Intercept calibration completed",
            adjustment=self.intercept_adjustment_,
        )

        return self

    def transform(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Apply intercept adjustment to predictions.

        Args:
            y_pred_proba: Predicted probabilities

        Returns:
            Calibrated probabilities

        Example:
            >>> y_calibrated = calibrator.transform(predictions_oot)
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted")

        return self._apply_adjustment(y_pred_proba, self.intercept_adjustment_)

    def fit_transform(
        self,
        y_pred_proba: np.ndarray,
        y_true: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels

        Returns:
            Calibrated probabilities

        Example:
            >>> y_calibrated = calibrator.fit_transform(predictions, y_true)
        """
        return self.fit(y_pred_proba, y_true).transform(y_pred_proba)

    def _apply_adjustment(
        self,
        y_pred_proba: np.ndarray,
        adjustment: float,
    ) -> np.ndarray:
        """Apply intercept adjustment in log-odds space."""
        # Convert to log-odds
        epsilon = 1e-10
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        log_odds = np.log(y_pred_proba / (1 - y_pred_proba))

        # Adjust
        adjusted_log_odds = log_odds + adjustment

        # Convert back to probability
        adjusted_proba = 1 / (1 + np.exp(-adjusted_log_odds))

        return adjusted_proba

    def get_adjustment(self) -> float:
        """
        Get intercept adjustment value.

        Returns:
            Adjustment value

        Example:
            >>> adjustment = calibrator.get_adjustment()
            >>> print(f"Intercept adjustment: {adjustment:.4f}")
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator not fitted")

        return self.intercept_adjustment_
