"""
SHAP-based model explainability for scorecards.

Uses SHAP (SHapley Additive exPlanations) to explain individual predictions
and overall model behavior.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based explainer for scorecard models.
    
    Provides both global and local explanations using SHAP values.
    
    Example:
        >>> explainer = SHAPExplainer(model)
        >>> explainer.fit(X_train)
        >>> shap_values = explainer.explain(X_test)
        >>> explainer.plot_summary(X_test)
    """
    
    def __init__(
        self,
        model: Any,
        model_type: str = "tree",
        use_kernel: bool = False,
    ) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            model: Fitted model (must have predict or predict_proba)
            model_type: Model type ('tree', 'linear', 'deep', 'kernel')
            use_kernel: Force use of KernelExplainer (model-agnostic)
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP not available. Install with: pip install shap"
            )
        
        self.model = model
        self.model_type = model_type
        self.use_kernel = use_kernel
        self.logger = get_audit_logger()
        
        self.explainer_: Optional[Any] = None
        self.feature_names_: Optional[List[str]] = None
        self.is_fitted_: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = 100,
    ) -> "SHAPExplainer":
        """
        Fit SHAP explainer on background data.
        
        Args:
            X: Background data for SHAP
            sample_size: Number of samples for background (None = all)
        
        Returns:
            Self
        """
        self.logger.info(
            "Fitting SHAP explainer",
            n_samples=len(X),
            model_type=self.model_type,
        )
        
        self.feature_names_ = list(X.columns)
        
        # Sample background data if needed
        if sample_size and len(X) > sample_size:
            background = X.sample(n=sample_size, random_state=42)
        else:
            background = X
        
        # Create appropriate explainer
        if self.use_kernel or self.model_type == "kernel":
            # Model-agnostic KernelExplainer
            self.explainer_ = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
        elif self.model_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)
        elif self.model_type == "linear":
            self.explainer_ = shap.LinearExplainer(
                self.model,
                background
            )
        else:
            # Fallback to Explainer (auto-detect)
            self.explainer_ = shap.Explainer(self.model, background)
        
        self.is_fitted_ = True
        self.logger.info("SHAP explainer fitted successfully")
        
        return self
    
    def explain(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Calculate SHAP values for data.
        
        Args:
            X: Data to explain
            check_additivity: Verify SHAP values sum to prediction
        
        Returns:
            SHAP values array (n_samples x n_features)
        """
        if not self.is_fitted_:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        self.logger.info("Calculating SHAP values", n_samples=len(X))
        
        shap_values = self.explainer_.shap_values(
            X,
            check_additivity=check_additivity
        )
        
        # Handle binary classification case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        return shap_values
    
    def explain_single(
        self,
        x: pd.Series,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Explain single prediction with top contributing features.
        
        Args:
            x: Single observation
            top_n: Number of top features to return
        
        Returns:
            DataFrame with features and SHAP values
        """
        X = pd.DataFrame([x])
        shap_values = self.explain(X)[0]
        
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'value': x.values,
            'shap_value': shap_values,
            'abs_shap': np.abs(shap_values),
        }).sort_values('abs_shap', ascending=False)
        
        return df.head(top_n)
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        method: str = "mean_abs",
    ) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            X: Data to analyze
            method: Aggregation method ('mean_abs', 'mean', 'max')
        
        Returns:
            DataFrame with features and importance scores
        """
        shap_values = self.explain(X)
        
        if method == "mean_abs":
            importance = np.abs(shap_values).mean(axis=0)
        elif method == "mean":
            importance = shap_values.mean(axis=0)
        elif method == "max":
            importance = np.abs(shap_values).max(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        return df
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = "dot",
        max_display: int = 20,
    ) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            X: Data to visualize
            plot_type: Plot type ('dot', 'bar', 'violin')
            max_display: Maximum features to display
        """
        shap_values = self.explain(X)
        
        shap.summary_plot(
            shap_values,
            X,
            plot_type=plot_type,
            max_display=max_display,
            show=True,
        )
    
    def plot_waterfall(
        self,
        x: pd.Series,
        max_display: int = 10,
    ) -> None:
        """
        Create waterfall plot for single prediction.
        
        Args:
            x: Single observation
            max_display: Maximum features to display
        """
        X = pd.DataFrame([x])
        shap_values = self.explain(X)
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer_.expected_value,
            data=X.iloc[0].values,
            feature_names=self.feature_names_,
        )
        
        shap.waterfall_plot(explanation, max_display=max_display, show=True)
    
    def plot_force(
        self,
        x: pd.Series,
        matplotlib: bool = False,
    ) -> None:
        """
        Create force plot for single prediction.
        
        Args:
            x: Single observation
            matplotlib: Use matplotlib instead of JavaScript
        """
        X = pd.DataFrame([x])
        shap_values = self.explain(X)
        
        shap.force_plot(
            self.explainer_.expected_value,
            shap_values[0],
            X.iloc[0],
            matplotlib=matplotlib,
            show=True,
        )
    
    def export_explanation(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Export SHAP explanations to dictionary.
        
        Args:
            X: Data to explain
        
        Returns:
            Dictionary with explanations
        """
        shap_values = self.explain(X)
        
        return {
            "shap_values": shap_values.tolist(),
            "feature_names": self.feature_names_,
            "expected_value": float(self.explainer_.expected_value),
            "feature_importance": self.get_feature_importance(X).to_dict('records'),
        }
