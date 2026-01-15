"""
Logistic regression scorecard with sample weighting support.

Supports weighted samples from compressed data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from cr_score.core.logging import get_audit_logger


class LogisticScorecard:
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
            penalty = "none"

        self.model_ = LogisticRegression(
            penalty=penalty,
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs" if penalty in ["none", "l2"] else "saga",
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
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary with performance metrics

        Example:
            >>> metrics = model.get_performance_metrics(y_test, predictions)
            >>> print(f"AUC: {metrics['auc']:.3f}")
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        # ROC AUC
        auc_score = roc_auc_score(y_true, y_pred_proba)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Gini coefficient
        gini = 2 * auc_score - 1

        # KS statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks = np.max(tpr - fpr)

        return {
            "auc": float(auc_score),
            "gini": float(gini),
            "ks": float(ks),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

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
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": float(auc(fpr, tpr)),
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
