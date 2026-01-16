"""
LightGBM scorecard for fast gradient boosting.

Efficient gradient boosting with histogram-based learning.
"""

from typing import Any, Dict, Optional

import pandas as pd

from cr_score.model.base import BaseScorecardModel

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LightGBMScorecard(BaseScorecardModel):
    """
    LightGBM classifier for scorecard development.
    
    Fast, distributed, high-performance gradient boosting framework.
    Handles categorical features natively and provides excellent speed.
    
    Example:
        >>> model = LightGBMScorecard(n_estimators=100, num_leaves=31)
        >>> model.fit(X_train, y_train, sample_weight=weights)
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> metrics = model.get_performance_metrics(y_test, y_pred)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize LightGBM scorecard model.
        
        Args:
            n_estimators: Number of boosting rounds
            num_leaves: Maximum number of leaves in one tree
            max_depth: Maximum depth of a tree (-1 = no limit)
            learning_rate: Boosting learning rate
            min_child_samples: Minimum data in one leaf
            min_child_weight: Minimum sum of hessian in one leaf
            subsample: Fraction of data for training each tree
            colsample_bytree: Fraction of features for training each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            scale_pos_weight: Weight of positive class (None = auto)
            random_state: Random seed
        
        Raises:
            ImportError: If lightgbm is not installed
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is not installed. "
                "Install with: pip install lightgbm"
            )
        
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
    
    def _create_model(self) -> "lgb.LGBMClassifier":
        """
        Create LightGBM model instance.
        
        Returns:
            LGBMClassifier instance
        """
        params = {
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        
        return lgb.LGBMClassifier(**params)
    
    def get_feature_importance(self, importance_type: str = "split") -> pd.DataFrame:
        """
        Get feature importance from LightGBM.
        
        Args:
            importance_type: Type of importance ('split' or 'gain')
        
        Returns:
            DataFrame with feature importance
        
        Example:
            >>> importance = model.get_feature_importance(importance_type='gain')
            >>> print(importance.head())
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model_.feature_importances_,
        }).sort_values("importance", ascending=False)
    
    def export_model(self) -> Dict[str, Any]:
        """
        Export model to dictionary.
        
        Returns:
            Dictionary with model parameters
        """
        base_export = super().export_model()
        base_export.update({
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
        })
        
        return base_export
