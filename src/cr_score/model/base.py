"""
Base class for all scorecard models.

Provides common interface and evaluation integration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from cr_score.core.logging import get_audit_logger
from cr_score.evaluation import PerformanceEvaluator


class BaseScorecardModel(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for all scorecard models.
    
    Provides common interface for model training, prediction, and evaluation.
    All concrete model implementations should inherit from this class.
    
    Attributes:
        model_: The underlying sklearn-compatible model
        feature_names_: Names of features used for training
        is_fitted_: Whether the model has been fitted
        logger: Audit logger
        evaluator: Performance evaluator for metrics
    """
    
    _estimator_type = "classifier"
    
    def __init__(self, random_state: int = 42):
        """
        Initialize base scorecard model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = get_audit_logger()
        self.evaluator = PerformanceEvaluator()
        
        self.model_: Optional[Any] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the underlying model instance.
        
        Must be implemented by subclasses.
        
        Returns:
            Model instance
        """
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> "BaseScorecardModel":
        """
        Fit the scorecard model.
        
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
            f"Fitting {self.__class__.__name__}",
            n_samples=len(X),
            n_features=X.shape[1],
            weighted=sample_weight is not None,
        )
        
        self.feature_names_ = list(X.columns)
        
        # Create model if not already created
        if self.model_ is None:
            self.model_ = self._create_model()
        
        # Fit model
        weights = sample_weight.values if sample_weight is not None else None
        self.model_.fit(X.values, y.values, sample_weight=weights)
        
        self.is_fitted_ = True
        
        self.logger.info(f"{self.__class__.__name__} training completed")
        
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
        
        Includes classification, ranking, calibration, and optionally stability metrics.
        
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
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Must be implemented by subclasses based on model type.
        
        Returns:
            DataFrame with feature importance
        """
        pass
    
    def export_model(self) -> Dict[str, Any]:
        """
        Export model to dictionary.
        
        Returns:
            Dictionary with model parameters
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        return {
            "model_type": self.__class__.__name__,
            "feature_names": self.feature_names_,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted_,
        }
