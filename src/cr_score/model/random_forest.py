"""
Random Forest scorecard for non-linear patterns.

Tree-based ensemble model with sample weighting support.
"""

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cr_score.model.base import BaseScorecardModel


class RandomForestScorecard(BaseScorecardModel):
    """
    Random Forest classifier for scorecard development.
    
    Handles non-linear relationships and feature interactions automatically.
    Provides feature importance based on tree splits.
    
    Example:
        >>> model = RandomForestScorecard(n_estimators=100, max_depth=5)
        >>> model.fit(X_train, y_train, sample_weight=weights)
        >>> y_pred = model.predict_proba(X_test)[:, 1]
        >>> metrics = model.get_performance_metrics(y_test, y_pred)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize Random Forest scorecard model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features for best split ('sqrt', 'log2', None)
            class_weight: Weights for classes ('balanced', None)
            random_state: Random seed
        """
        super().__init__(random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
    
    def _create_model(self) -> RandomForestClassifier:
        """
        Create Random Forest model instance.
        
        Returns:
            RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        
        Based on mean decrease in impurity (Gini importance).
        
        Returns:
            DataFrame with feature importance
        
        Example:
            >>> importance = model.get_feature_importance()
            >>> print(importance.head())
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.model_.feature_importances_,
        }).sort_values("importance", ascending=False)
    
    def get_tree_depths(self) -> Dict[str, Any]:
        """
        Get statistics about tree depths in the forest.
        
        Returns:
            Dictionary with depth statistics
        
        Example:
            >>> depths = model.get_tree_depths()
            >>> print(f"Average depth: {depths['mean']:.1f}")
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        
        import numpy as np
        
        depths = [tree.get_depth() for tree in self.model_.estimators_]
        
        return {
            "min": int(np.min(depths)),
            "max": int(np.max(depths)),
            "mean": float(np.mean(depths)),
            "median": float(np.median(depths)),
        }
    
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
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
        })
        
        if self.is_fitted_:
            base_export["tree_depths"] = self.get_tree_depths()
        
        return base_export
