"""
Univariate analysis for EDA.

Generate statistics and distributions for individual variables.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class UnivariateAnalyzer:
    """
    Univariate statistical analysis.

    Generates comprehensive statistics for numeric and categorical variables.

    Example:
        >>> analyzer = UnivariateAnalyzer()
        >>> stats = analyzer.analyze(df, target_col="default")
        >>> print(stats["age"]["mean"])
    """

    def __init__(self) -> None:
        """Initialize univariate analyzer."""
        self.logger = get_audit_logger()

    def analyze(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform univariate analysis on all features.

        Args:
            df: Input DataFrame
            target_col: Target variable (for event rate calculation)
            features: List of features to analyze (None = all except target)

        Returns:
            Dictionary mapping feature name to statistics dict

        Example:
            >>> stats = analyzer.analyze(df, target_col="default")
            >>> for feat, feat_stats in stats.items():
            ...     print(f"{feat}: missing={feat_stats['missing_pct']:.1f}%")
        """
        if features is None:
            features = [col for col in df.columns if col != target_col]

        self.logger.info(f"Starting univariate analysis on {len(features)} features")

        results: Dict[str, Dict[str, Any]] = {}

        for feature in features:
            if feature not in df.columns:
                continue

            results[feature] = self._analyze_feature(df, feature, target_col)

        self.logger.info("Univariate analysis completed")

        return results

    def _analyze_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze single feature."""
        series = df[feature]
        stats: Dict[str, Any] = {
            "feature_name": feature,
            "dtype": str(series.dtype),
            "count": int(len(series)),
            "missing_count": int(series.isnull().sum()),
            "missing_pct": float(series.isnull().sum() / len(series) * 100),
            "unique_count": int(series.nunique()),
            "unique_ratio": float(series.nunique() / len(series)),
        }

        if pd.api.types.is_numeric_dtype(series):
            stats.update(self._numeric_stats(series))
        else:
            stats.update(self._categorical_stats(series))

        # Event rate by feature (if target provided)
        if target_col and target_col in df.columns:
            stats["event_rates"] = self._calculate_event_rates(df, feature, target_col)

        return stats

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate numeric variable statistics."""
        return {
            "variable_type": "numeric",
            "mean": float(series.mean()) if not series.isnull().all() else None,
            "std": float(series.std()) if not series.isnull().all() else None,
            "min": float(series.min()) if not series.isnull().all() else None,
            "q25": float(series.quantile(0.25)) if not series.isnull().all() else None,
            "median": float(series.median()) if not series.isnull().all() else None,
            "q75": float(series.quantile(0.75)) if not series.isnull().all() else None,
            "max": float(series.max()) if not series.isnull().all() else None,
            "skewness": float(series.skew()) if not series.isnull().all() else None,
            "kurtosis": float(series.kurtosis()) if not series.isnull().all() else None,
        }

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate categorical variable statistics."""
        value_counts = series.value_counts()
        top_5 = value_counts.head(5)

        return {
            "variable_type": "categorical",
            "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
            "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "mode_pct": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
            "top_5_values": top_5.index.tolist(),
            "top_5_counts": top_5.values.tolist(),
            "top_5_pcts": (top_5 / len(series) * 100).tolist(),
        }

    def _calculate_event_rates(
        self,
        df: pd.DataFrame,
        feature: str,
        target_col: str,
    ) -> Dict[str, Any]:
        """Calculate event rates by feature values."""
        grouped = df.groupby(feature, dropna=False)[target_col].agg(
            total_count="count",
            event_count="sum",
        )

        grouped["event_rate"] = grouped["event_count"] / grouped["total_count"]

        return {
            "overall_event_rate": float(df[target_col].mean()),
            "by_value": grouped.to_dict("index"),
        }

    def generate_summary_table(
        self,
        stats: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Generate summary table from univariate statistics.

        Args:
            stats: Statistics dictionary from analyze()

        Returns:
            DataFrame with one row per feature

        Example:
            >>> stats = analyzer.analyze(df)
            >>> summary = analyzer.generate_summary_table(stats)
            >>> print(summary[["feature_name", "dtype", "missing_pct", "unique_count"]])
        """
        rows = []

        for feature, feature_stats in stats.items():
            row = {
                "feature_name": feature_stats["feature_name"],
                "dtype": feature_stats["dtype"],
                "variable_type": feature_stats.get("variable_type", "unknown"),
                "count": feature_stats["count"],
                "missing_count": feature_stats["missing_count"],
                "missing_pct": feature_stats["missing_pct"],
                "unique_count": feature_stats["unique_count"],
                "unique_ratio": feature_stats["unique_ratio"],
            }

            if feature_stats.get("variable_type") == "numeric":
                row.update({
                    "mean": feature_stats.get("mean"),
                    "std": feature_stats.get("std"),
                    "min": feature_stats.get("min"),
                    "median": feature_stats.get("median"),
                    "max": feature_stats.get("max"),
                })
            elif feature_stats.get("variable_type") == "categorical":
                row.update({
                    "mode": feature_stats.get("mode"),
                    "mode_pct": feature_stats.get("mode_pct"),
                })

            rows.append(row)

        return pd.DataFrame(rows)
