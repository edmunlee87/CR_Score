"""
XGBoost scorecard for gradient boosting.

High-performance gradient boosting with regularization.
"""

from typing import Any, Dict, Optional

import pandas as pd

from cr_score.model.base import BaseScorecardModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostScorecard(BaseScorecardModel):
    """
    XGBoost classifier for scorecard development.
    
    Gradient boosting with built-in regularization and handling of missing values.
    Provides high performance and feature importance.
    
    Example:
        >>> model = XGBoostScorecard(n_estimators=100, max_depth=5)
        >>> model.fit(X_train, y_train, sample_weight=weights)
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> metrics = model.get_performance_metrics(y_test, y_pred)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """
        Initialize XGBoost scorecard model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of a tree
            learning_rate: Step size shrinkage (eta)
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            subsample: Fraction of samples for training each tree
            colsample_bytree: Fraction of features for training each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            scale_pos_weight: Balancing of positive and negative weights
            random_state: Random seed
        
        Raises:
            ImportError: If xgboost is not installed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. "
                "Install with: pip install xgboost"
            )
        
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
    
    def _create_model(self) -> "xgb.XGBClassifier":
        """
        Create XGBoost model instance.
        
        Returns:
            XGBClassifier instance
        """
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    
    def get_feature_importance(self, importance_type: str = "weight") -> pd.DataFrame:
        """
        Get feature importance from XGBoost.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
        
        Returns:
            DataFrame with feature importance
        
        Example:
            >>> importance = model.get_feature_importance(importance_type='gain')
            >>> print(importance.head())
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        # Get importance scores
        importance_dict = self.model_.get_booster().get_score(importance_type=importance_type)
        
        # Map feature names
        importance_data = []
        for i, feature_name in enumerate(self.feature_names_):
            feat_key = f"f{i}"
            importance_data.append({
                "feature": feature_name,
                "importance": importance_dict.get(feat_key, 0.0),
            })
        
        return pd.DataFrame(importance_data).sort_values("importance", ascending=False)
    
    def export_model(self) -> Dict[str, Any]:
        """
        Export model to dictionary.
        
        Returns:
            Dictionary with model parameters
        """
        base_export = super().export_model()
        base_export.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
        })
        
        return base_export
