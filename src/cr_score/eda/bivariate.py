"""
Bivariate analysis for feature-target relationships.

Calculate correlations, IV (Information Value), and predictive metrics.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from cr_score.core.logging import get_audit_logger


class BivariateAnalyzer:
    """
    Bivariate analysis between features and target.

    Calculates correlations, statistical tests, and preliminary IV estimates.

    Example:
        >>> analyzer = BivariateAnalyzer()
        >>> results = analyzer.analyze(df, target_col="default")
        >>> top_features = results[results["correlation_abs"] > 0.1]
    """

    def __init__(self) -> None:
        """Initialize bivariate analyzer."""
        self.logger = get_audit_logger()

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: str,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform bivariate analysis for all features vs target.

        Args:
            df: Input DataFrame
            target_col: Target variable
            features: Features to analyze (None = all except target)

        Returns:
            DataFrame with bivariate statistics per feature

        Example:
            >>> results = analyzer.analyze(df, target_col="default")
            >>> print(results[["feature", "correlation", "p_value", "iv_estimate"]])
        """
        if features is None:
            features = [col for col in df.columns if col != target_col]

        self.logger.info(
            f"Starting bivariate analysis on {len(features)} features vs {target_col}"
        )

        results = []

        for feature in features:
            if feature not in df.columns:
                continue

            feature_results = self._analyze_feature_target(df, feature, target_col)
            if feature_results:
                results.append(feature_results)

        df_results = pd.DataFrame(results)

        self.logger.info("Bivariate analysis completed")

        return df_results

    def _analyze_feature_target(
        self,
        df: pd.DataFrame,
        feature: str,
        target_col: str,
    ) -> Optional[Dict]:
        """Analyze single feature vs target."""
        # Remove missing values
        valid_idx = df[[feature, target_col]].dropna().index
        if len(valid_idx) < 10:
            return None

        feature_series = df.loc[valid_idx, feature]
        target_series = df.loc[valid_idx, target_col]

        results = {
            "feature": feature,
            "dtype": str(df[feature].dtype),
            "valid_count": len(valid_idx),
        }

        # Numeric analysis
        if pd.api.types.is_numeric_dtype(feature_series):
            results.update(
                self._numeric_vs_target(feature_series, target_series)
            )
        else:
            results.update(
                self._categorical_vs_target(feature_series, target_series)
            )

        return results

    def _numeric_vs_target(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Dict:
        """Analyze numeric feature vs binary target."""
        # Pearson correlation
        correlation, p_value = stats.pearsonr(feature, target)

        # Point-biserial correlation (appropriate for continuous vs binary)
        point_biserial, pb_pvalue = stats.pointbiserialr(feature, target)

        # Mean difference test (t-test)
        group_0 = feature[target == 0]
        group_1 = feature[target == 1]

        if len(group_0) > 0 and len(group_1) > 0:
            t_stat, t_pvalue = stats.ttest_ind(group_1, group_0)
            mean_diff = group_1.mean() - group_0.mean()
        else:
            t_stat, t_pvalue, mean_diff = None, None, None

        return {
            "variable_type": "numeric",
            "correlation": float(correlation),
            "correlation_abs": float(abs(correlation)),
            "p_value": float(p_value),
            "point_biserial": float(point_biserial),
            "t_statistic": float(t_stat) if t_stat is not None else None,
            "t_pvalue": float(t_pvalue) if t_pvalue is not None else None,
            "mean_diff": float(mean_diff) if mean_diff is not None else None,
        }

    def _categorical_vs_target(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> Dict:
        """Analyze categorical feature vs binary target."""
        # Chi-square test
        contingency_table = pd.crosstab(feature, target)

        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            chi2, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
        else:
            chi2, chi2_pvalue = None, None

        # Cramér's V (effect size for chi-square)
        cramers_v = self._calculate_cramers_v(contingency_table) if chi2 is not None else None

        return {
            "variable_type": "categorical",
            "chi2_statistic": float(chi2) if chi2 is not None else None,
            "chi2_pvalue": float(chi2_pvalue) if chi2_pvalue is not None else None,
            "cramers_v": float(cramers_v) if cramers_v is not None else None,
            "correlation": None,
            "correlation_abs": float(cramers_v) if cramers_v is not None else None,
            "p_value": float(chi2_pvalue) if chi2_pvalue is not None else None,
        }

    def _calculate_cramers_v(self, contingency_table: pd.DataFrame) -> float:
        """Calculate Cramér's V effect size."""
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1

        if min_dim > 0:
            return np.sqrt(chi2 / (n * min_dim))
        else:
            return 0.0

    def rank_features(
        self,
        bivariate_results: pd.DataFrame,
        by: str = "correlation_abs",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Rank features by predictive strength.

        Args:
            bivariate_results: Results from analyze()
            by: Column to sort by (correlation_abs, chi2_statistic, etc.)
            ascending: Sort order

        Returns:
            Sorted DataFrame

        Example:
            >>> ranked = analyzer.rank_features(results, by="correlation_abs")
            >>> print(ranked.head(10))
        """
        return bivariate_results.sort_values(by=by, ascending=ascending)
