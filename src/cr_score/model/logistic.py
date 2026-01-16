"""
Logistic regression scorecard with sample weighting support.

Supports weighted samples from compressed data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cr_score.core.logging import get_audit_logger
from cr_score.evaluation import PerformanceEvaluator


class LogisticScorecard(BaseEstimator, ClassifierMixin):
    """
    Logistic regression model for scorecard development.

    Supports sample weighting for compressed data and generates
    comprehensive diagnostics.

    Example:
        >>> model = LogisticScorecard(regularization="l2", C=1.0)
        >>> model.fit(X_train, y_train, sample_weight=weights)
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> metrics = model.get_performance_metrics(y_test, y_pred)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        regularization: Optional[str] = None,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """
        Initialize logistic scorecard model.

        Args:
            regularization: Regularization type (None, "l1", "l2", "elasticnet")
            C: Inverse regularization strength (smaller = stronger)
            random_state: Random seed
        """
        self.regularization = regularization
        self.C = C
        self.random_state = random_state
        self.logger = get_audit_logger()
        self.evaluator = PerformanceEvaluator()

        self.model_: Optional[LogisticRegression] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "LogisticScorecard":
        """
        Fit logistic regression model.

        Args:
            X: Feature matrix (WoE-encoded features)
            y: Target variable (binary)
            sample_weight: Sample weights (for compressed data)

        Returns:
            Self

        Example:
            >>> model.fit(X_train_woe, y_train, sample_weight=train_weights)
        """
        self.logger.info(
            "Fitting logistic regression model",
            n_samples=len(X),
            n_features=X.shape[1],
            weighted=sample_weight is not None,
        )

        self.feature_names_ = list(X.columns)

        # Configure model
        penalty = self.regularization if self.regularization else "l2"
        if self.regularization is None:
            penalty = None

        self.model_ = LogisticRegression(
            penalty=penalty,
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs" if penalty in [None, "l2"] else "saga",
        )

        # Fit model
        weights = sample_weight.values if sample_weight is not None else None
        self.model_.fit(X.values, y.values, sample_weight=weights)

        self.is_fitted_ = True

        self.logger.info(
            "Model training completed",
            intercept=float(self.model_.intercept_[0]),
            n_features_used=int((self.model_.coef_[0] != 0).sum()),
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities (shape: n_samples x 2)

        Example:
            >>> probas = model.predict_proba(X_test)
            >>> default_probas = probas[:, 1]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")

        return self.model_.predict_proba(X.values)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Array of predicted labels

        Example:
            >>> predictions = model.predict(X_test, threshold=0.3)
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients.

        Returns:
            DataFrame with features and coefficients

        Example:
            >>> coefs = model.get_coefficients()
            >>> print(coefs.sort_values("coefficient", ascending=False))
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")

        return pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": self.model_.coef_[0],
            "abs_coefficient": np.abs(self.model_.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)

    def get_performance_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        include_stability: bool = False,
        y_train_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Now includes classification, ranking, calibration, and optionally stability metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            include_stability: Whether to include PSI/CSI stability metrics
            y_train_proba: Training probabilities for stability comparison (optional)

        Returns:
            Dictionary with all performance metrics

        Example:
            >>> metrics = model.get_performance_metrics(y_test, predictions)
            >>> print(f"AUC: {metrics['ranking']['auc']:.3f}")
            >>> print(f"Brier Score: {metrics['calibration']['brier_score']:.3f}")
            >>> print(f"MCC: {metrics['classification']['mcc']:.3f}")
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else y_true

        # Get all metrics using the unified evaluator
        results = self.evaluator.evaluate_all(
            y_true=y_true_np,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            threshold=threshold,
        )

        # Add stability metrics if requested
        if include_stability and y_train_proba is not None:
            stability_results = self.evaluator.evaluate_stability(
                expected=y_train_proba,
                actual=y_pred_proba,
            )
            results["stability"] = stability_results

        # Add interpretations
        results["interpretations"] = self.evaluator.interpret_results(results)

        return results

    def get_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with FPR, TPR, and thresholds

        Example:
            >>> roc_data = model.get_roc_curve(y_test, predictions)
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(roc_data["fpr"], roc_data["tpr"])
        """
        from sklearn.metrics import auc as sklearn_auc, roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(sklearn_auc(fpr, tpr)),
        }

    def get_metrics_summary(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Get summary table of key metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            DataFrame with key metrics

        Example:
            >>> summary = model.get_metrics_summary(y_test, predictions)
            >>> print(summary)
        """
        metrics = self.get_performance_metrics(y_true, y_pred_proba, threshold)
        return self.evaluator.summary(metrics)

    def get_lift_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate lift curve data.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins/deciles

        Returns:
            DataFrame with lift curve data

        Example:
            >>> lift = model.get_lift_curve(y_test, predictions, n_bins=10)
            >>> print(lift)
        """
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else y_true
        return self.evaluator.ranking.calculate_lift_curve(y_true_np, y_pred_proba, n_bins)

    def get_gains_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate cumulative gains curve data.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins/deciles

        Returns:
            DataFrame with gains curve data

        Example:
            >>> gains = model.get_gains_curve(y_test, predictions, n_bins=10)
            >>> print(gains)
        """
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else y_true
        return self.evaluator.ranking.calculate_gains_curve(y_true_np, y_pred_proba, n_bins)

    def get_calibration_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate calibration curve data.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            Dictionary with calibration curve data

        Example:
            >>> calib = model.get_calibration_curve(y_test, predictions)
            >>> plt.plot(calib['mean_predicted'], calib['fraction_positives'])
        """
        y_true_np = y_true.values if isinstance(y_true, pd.Series) else y_true
        frac_pos, mean_pred = self.evaluator.calibration.calculate_calibration_curve(
            y_true_np, y_pred_proba, n_bins
        )

        return {
            "fraction_positives": frac_pos,
            "mean_predicted": mean_pred,
        }

    def calculate_psi(
        self,
        train_proba: np.ndarray,
        test_proba: np.ndarray,
        bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI).

        Args:
            train_proba: Training probabilities (baseline)
            test_proba: Test probabilities (current)
            bins: Number of bins

        Returns:
            Dictionary with PSI value and interpretation

        Example:
            >>> psi_result = model.calculate_psi(train_probas, test_probas)
            >>> print(f"PSI: {psi_result['psi']:.3f} ({psi_result['status']})")
        """
        psi = self.evaluator.stability.calculate_psi(train_proba, test_proba, bins)
        status = self.evaluator.stability.psi_interpretation(psi)

        return {
            "psi": psi,
            "status": status,
            "breakdown": self.evaluator.stability.calculate_psi_breakdown(
                train_proba, test_proba, bins
            ),
        }

    def export_model(self) -> Dict[str, Any]:
        """
        Export model to dictionary.

        Returns:
            Dictionary with model parameters and coefficients

        Example:
            >>> model_dict = model.export_model()
            >>> import json
            >>> with open("model.json", "w") as f:
            ...     json.dump(model_dict, f)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")

        return {
            "model_type": "logistic_regression",
            "intercept": float(self.model_.intercept_[0]),
            "coefficients": {
                feat: float(coef)
                for feat, coef in zip(self.feature_names_, self.model_.coef_[0])
            },
            "feature_names": self.feature_names_,
            "regularization": self.regularization,
            "C": self.C,
        }
