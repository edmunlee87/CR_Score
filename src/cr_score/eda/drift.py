"""
Drift analysis for monitoring data distribution changes.

Implements PSI (Population Stability Index) and CSI (Characteristic Stability Index).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class DriftAnalyzer:
    """
    Analyze drift between baseline and comparison datasets.

    Calculates PSI (Population Stability Index) for detecting distribution shifts.

    Example:
        >>> analyzer = DriftAnalyzer()
        >>> psi_results = analyzer.calculate_psi_all(
        ...     df_baseline,
        ...     df_comparison,
        ...     features=["age", "income", "credit_score"]
        ... )
        >>> high_drift = [f for f, psi in psi_results.items() if psi > 0.25]
    """

    PSI_THRESHOLDS = {
        "no_change": 0.1,
        "moderate_change": 0.25,
    }

    def __init__(self) -> None:
        """Initialize drift analyzer."""
        self.logger = get_audit_logger()

    def calculate_psi_all(
        self,
        df_baseline: pd.DataFrame,
        df_comparison: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate PSI for all features.

        Args:
            df_baseline: Baseline (development) dataset
            df_comparison: Comparison (OOT/validation) dataset
            features: Features to analyze (None = all common columns)
            n_bins: Number of bins for numeric variables

        Returns:
            Dictionary mapping feature name to PSI value

        Example:
            >>> psi_scores = analyzer.calculate_psi_all(df_dev, df_oot)
            >>> for feat, psi in psi_scores.items():
            ...     status = "ALERT" if psi > 0.25 else "OK"
            ...     print(f"{feat}: PSI={psi:.3f} [{status}]")
        """
        if features is None:
            features = list(set(df_baseline.columns).intersection(df_comparison.columns))

        self.logger.info(f"Calculating PSI for {len(features)} features")

        psi_results = {}

        for feature in features:
            if feature not in df_baseline.columns or feature not in df_comparison.columns:
                continue

            psi = self.calculate_psi(
                df_baseline[feature],
                df_comparison[feature],
                n_bins=n_bins,
            )

            psi_results[feature] = psi

        self.logger.info(
            "PSI calculation completed",
            features_analyzed=len(psi_results),
            high_drift_count=sum(1 for psi in psi_results.values() if psi > self.PSI_THRESHOLDS["moderate_change"]),
        )

        return psi_results

    def calculate_psi(
        self,
        baseline: pd.Series,
        comparison: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate PSI for single variable.

        PSI Formula:
        PSI = Σ [(% comparison - % baseline) * ln(% comparison / % baseline)]

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.25: Moderate change
        - PSI ≥ 0.25: Significant change (investigate!)

        Args:
            baseline: Baseline distribution
            comparison: Comparison distribution
            n_bins: Number of bins for numeric variables

        Returns:
            PSI value

        Example:
            >>> psi = analyzer.calculate_psi(df_dev["age"], df_oot["age"])
            >>> print(f"PSI: {psi:.3f}")
        """
        # Remove missing values
        baseline_clean = baseline.dropna()
        comparison_clean = comparison.dropna()

        if len(baseline_clean) == 0 or len(comparison_clean) == 0:
            return np.nan

        # Determine if numeric or categorical
        if pd.api.types.is_numeric_dtype(baseline):
            return self._calculate_psi_numeric(baseline_clean, comparison_clean, n_bins)
        else:
            return self._calculate_psi_categorical(baseline_clean, comparison_clean)

    def _calculate_psi_numeric(
        self,
        baseline: pd.Series,
        comparison: pd.Series,
        n_bins: int,
    ) -> float:
        """Calculate PSI for numeric variable."""
        # Create bins based on baseline quantiles
        _, bin_edges = pd.qcut(baseline, q=n_bins, retbins=True, duplicates="drop")

        # Ensure edges cover full range
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Bin both distributions
        baseline_binned = pd.cut(baseline, bins=bin_edges)
        comparison_binned = pd.cut(comparison, bins=bin_edges)

        # Calculate distributions
        baseline_dist = baseline_binned.value_counts(normalize=True, sort=False)
        comparison_dist = comparison_binned.value_counts(normalize=True, sort=False)

        # Align indices
        all_bins = baseline_dist.index.union(comparison_dist.index)
        baseline_dist = baseline_dist.reindex(all_bins, fill_value=0)
        comparison_dist = comparison_dist.reindex(all_bins, fill_value=0)

        # Calculate PSI
        return self._psi_formula(baseline_dist.values, comparison_dist.values)

    def _calculate_psi_categorical(
        self,
        baseline: pd.Series,
        comparison: pd.Series,
    ) -> float:
        """Calculate PSI for categorical variable."""
        # Calculate distributions
        baseline_dist = baseline.value_counts(normalize=True)
        comparison_dist = comparison.value_counts(normalize=True)

        # Align indices
        all_categories = baseline_dist.index.union(comparison_dist.index)
        baseline_dist = baseline_dist.reindex(all_categories, fill_value=0)
        comparison_dist = comparison_dist.reindex(all_categories, fill_value=0)

        # Calculate PSI
        return self._psi_formula(baseline_dist.values, comparison_dist.values)

    def _psi_formula(
        self,
        baseline_pcts: np.ndarray,
        comparison_pcts: np.ndarray,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Apply PSI formula.

        Args:
            baseline_pcts: Baseline percentages
            comparison_pcts: Comparison percentages
            epsilon: Small constant to avoid log(0)

        Returns:
            PSI value
        """
        # Add epsilon to avoid division by zero
        baseline_pcts = baseline_pcts + epsilon
        comparison_pcts = comparison_pcts + epsilon

        # PSI = Σ [(% comparison - % baseline) * ln(% comparison / % baseline)]
        psi = np.sum(
            (comparison_pcts - baseline_pcts) * np.log(comparison_pcts / baseline_pcts)
        )

        return float(psi)

    def generate_psi_report(
        self,
        psi_results: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Generate PSI report DataFrame.

        Args:
            psi_results: PSI results from calculate_psi_all()

        Returns:
            DataFrame with PSI values and drift status

        Example:
            >>> report = analyzer.generate_psi_report(psi_results)
            >>> print(report[report["drift_status"] == "Significant Change"])
        """
        rows = []

        for feature, psi in psi_results.items():
            if np.isnan(psi):
                status = "Unable to Calculate"
            elif psi < self.PSI_THRESHOLDS["no_change"]:
                status = "No Change"
            elif psi < self.PSI_THRESHOLDS["moderate_change"]:
                status = "Moderate Change"
            else:
                status = "Significant Change"

            rows.append({
                "feature": feature,
                "psi": psi,
                "drift_status": status,
                "requires_investigation": psi >= self.PSI_THRESHOLDS["moderate_change"],
            })

        df = pd.DataFrame(rows)
        return df.sort_values("psi", ascending=False)

    def calculate_csi(
        self,
        baseline_event_rates: pd.Series,
        comparison_event_rates: pd.Series,
    ) -> float:
        """
        Calculate CSI (Characteristic Stability Index).

        Similar to PSI but for event rates rather than distributions.

        Args:
            baseline_event_rates: Baseline event rates by segment
            comparison_event_rates: Comparison event rates by segment

        Returns:
            CSI value

        Example:
            >>> csi = analyzer.calculate_csi(
            ...     baseline_rates,
            ...     comparison_rates
            ... )
        """
        # Align indices
        all_segments = baseline_event_rates.index.union(comparison_event_rates.index)
        baseline_rates = baseline_event_rates.reindex(all_segments, fill_value=0)
        comparison_rates = comparison_event_rates.reindex(all_segments, fill_value=0)

        # CSI uses same formula as PSI
        return self._psi_formula(baseline_rates.values, comparison_rates.values)
