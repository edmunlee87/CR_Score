"""
Calibration metrics for assessing probability quality.

Brier score, log loss, calibration curves, and Hosmer-Lemeshow test.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


class CalibrationMetrics:
    """
    Calibration metrics for probability predictions.
    
    Assesses how well predicted probabilities match actual outcomes.
    
    Example:
        >>> metrics = CalibrationMetrics()
        >>> brier = metrics.calculate_brier_score(y_true, y_proba)
        >>> print(f"Brier Score: {brier:.3f}")
    """
    
    @staticmethod
    def calculate_brier_score(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate Brier score (mean squared error of probabilities).
        
        Lower is better. Range: [0, 1]
        - 0: Perfect calibration
        - 0.25: Baseline (random predictions)
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Brier score
        
        Example:
            >>> brier = CalibrationMetrics.calculate_brier_score(y_true, y_proba)
        """
        return float(brier_score_loss(y_true, y_proba))
    
    @staticmethod
    def calculate_log_loss(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate log loss (cross-entropy loss).
        
        Lower is better. Range: [0, inf]
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Log loss
        
        Example:
            >>> logloss = CalibrationMetrics.calculate_log_loss(y_true, y_proba)
        """
        return float(log_loss(y_true, y_proba))
    
    @staticmethod
    def calculate_calibration_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve data.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            strategy: Binning strategy ('uniform' or 'quantile')
        
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        
        Example:
            >>> frac_pos, mean_pred = CalibrationMetrics.calculate_calibration_curve(
            ...     y_true, y_proba, n_bins=10
            ... )
        """
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy=strategy
        )
        return prob_true, prob_pred
    
    @staticmethod
    def hosmer_lemeshow_test(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.
        
        Tests if predicted probabilities match observed frequencies.
        
        Interpretation:
        - p-value > 0.05: Good calibration (cannot reject null hypothesis)
        - p-value <= 0.05: Poor calibration (reject null hypothesis)
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            Dictionary with chi-square statistic and p-value
        
        Example:
            >>> hl_test = CalibrationMetrics.hosmer_lemeshow_test(y_true, y_proba)
            >>> print(f"H-L p-value: {hl_test['pvalue']:.4f}")
        """
        from scipy.stats import chi2
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins=bins, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate observed and expected in each bin
        chi_sq = 0.0
        df = 0
        
        for i in range(n_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            n = np.sum(mask)
            observed_pos = np.sum(y_true[mask])
            observed_neg = n - observed_pos
            
            expected_pos = np.sum(y_proba[mask])
            expected_neg = n - expected_pos
            
            # Avoid division by zero
            if expected_pos > 0 and expected_neg > 0:
                chi_sq += (observed_pos - expected_pos) ** 2 / expected_pos
                chi_sq += (observed_neg - expected_neg) ** 2 / expected_neg
                df += 1
        
        # Degrees of freedom = number of bins - 2
        df = max(df - 2, 1)
        
        # Calculate p-value
        pvalue = 1 - chi2.cdf(chi_sq, df)
        
        return {
            "chi_square": float(chi_sq),
            "df": int(df),
            "pvalue": float(pvalue),
            "well_calibrated": pvalue > 0.05,
        }
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted and actual probabilities
        across bins, weighted by the number of samples in each bin.
        
        Lower is better. Range: [0, 1]
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            ECE value
        
        Example:
            >>> ece = CalibrationMetrics.expected_calibration_error(y_true, y_proba)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins=bins, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        
        for i in range(n_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            n = np.sum(mask)
            confidence = np.mean(y_proba[mask])
            accuracy = np.mean(y_true[mask])
            
            ece += (n / len(y_true)) * np.abs(confidence - accuracy)
        
        return float(ece)
    
    @staticmethod
    def maximum_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).
        
        MCE is the maximum difference between predicted and actual probabilities
        across all bins.
        
        Lower is better. Range: [0, 1]
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            MCE value
        
        Example:
            >>> mce = CalibrationMetrics.maximum_calibration_error(y_true, y_proba)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins=bins, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        max_error = 0.0
        
        for i in range(n_bins):
            mask = bin_indices == i
            
            if not np.any(mask):
                continue
            
            confidence = np.mean(y_proba[mask])
            accuracy = np.mean(y_true[mask])
            
            error = np.abs(confidence - accuracy)
            max_error = max(max_error, error)
        
        return float(max_error)
    
    @staticmethod
    def calculate_all_calibration_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate all calibration metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
        
        Returns:
            Dictionary with all calibration metrics
        
        Example:
            >>> metrics = CalibrationMetrics.calculate_all_calibration_metrics(y_true, y_proba)
        """
        brier = CalibrationMetrics.calculate_brier_score(y_true, y_proba)
        logloss = CalibrationMetrics.calculate_log_loss(y_true, y_proba)
        ece = CalibrationMetrics.expected_calibration_error(y_true, y_proba, n_bins)
        mce = CalibrationMetrics.maximum_calibration_error(y_true, y_proba, n_bins)
        hl_test = CalibrationMetrics.hosmer_lemeshow_test(y_true, y_proba, n_bins)
        
        return {
            "brier_score": brier,
            "log_loss": logloss,
            "ece": ece,
            "mce": mce,
            "hosmer_lemeshow_chi_square": hl_test["chi_square"],
            "hosmer_lemeshow_pvalue": hl_test["pvalue"],
            "hosmer_lemeshow_well_calibrated": hl_test["well_calibrated"],
        }
    
    @staticmethod
    def calibration_slope_intercept(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate calibration slope and intercept.
        
        Fits a logistic regression: logit(y_true) ~ logit(y_proba)
        
        Interpretation:
        - Perfect calibration: slope = 1, intercept = 0
        - Slope < 1: Over-confident predictions
        - Slope > 1: Under-confident predictions
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Dictionary with slope and intercept
        
        Example:
            >>> calib = CalibrationMetrics.calibration_slope_intercept(y_true, y_proba)
        """
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to log-odds
        y_proba_clipped = np.clip(y_proba, 1e-15, 1 - 1e-15)
        logit_proba = np.log(y_proba_clipped / (1 - y_proba_clipped))
        
        # Fit logistic regression
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(logit_proba.reshape(-1, 1), y_true)
        
        return {
            "slope": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
        }
