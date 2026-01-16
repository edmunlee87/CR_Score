"""
Logistic regression scorecard with sample weighting support.

Supports weighted samples from compressed data.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cr_score.model.base import BaseScorecardModel


class LogisticScorecard(BaseScorecardModel):
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
        super().__init__(random_state=random_state)
        self.regularization = regularization
        self.C = C
    
    def _create_model(self) -> LogisticRegression:
        """
        Create logistic regression model instance.
        
        Returns:
            LogisticRegression instance
        """
        penalty = self.regularization if self.regularization else "l2"
        if self.regularization is None:
            penalty = None
        
        return LogisticRegression(
            penalty=penalty,
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs" if penalty in [None, "l2"] else "saga",
        )


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
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from coefficients.
        
        For logistic regression, absolute coefficients represent importance.
        
        Returns:
            DataFrame with feature importance
        
        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance.head())
        """
        return self.get_coefficients()


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
