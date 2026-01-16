"""
Model explainability and interpretability for scorecards.

Provides SHAP values, feature importance, contribution analysis,
and reason codes for individual predictions.
"""

from cr_score.explainability.shap_explainer import SHAPExplainer
from cr_score.explainability.reason_codes import ReasonCodeGenerator
from cr_score.explainability.feature_importance import FeatureImportanceAnalyzer

__all__ = [
    "SHAPExplainer",
    "ReasonCodeGenerator",
    "FeatureImportanceAnalyzer",
]
