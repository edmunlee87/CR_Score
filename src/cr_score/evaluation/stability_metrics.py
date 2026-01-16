"""
Stability metrics for monitoring model and data drift.

PSI (Population Stability Index) and CSI (Characteristic Stability Index).
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class StabilityMetrics:
    """
    Calculate stability metrics for model monitoring.
    
    Provides PSI, CSI, and other stability measures to detect drift.
    
    Example:
        >>> metrics = StabilityMetrics()
        >>> psi = metrics.calculate_psi(train_scores, test_scores)
        >>> print(f"PSI: {psi:.3f}")
    """
    
    @staticmethod
    def calculate_psi(
        expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        bins: Union[int, List[float]] = 10,
        epsilon: float = 0.0001,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in population distribution between two samples.
        
        Interpretation:
        - PSI < 0.1: No significant population change
        - 0.1 <= PSI < 0.2: Moderate population change
        - PSI >= 0.2: Significant population change (investigate)
        
        Args:
            expected: Reference/baseline distribution (e.g., training data)
            actual: Current distribution (e.g., production data)
            bins: Number of bins or explicit bin edges
            epsilon: Small constant to avoid log(0)
        
        Returns:
            PSI value
        
        Example:
            >>> psi = StabilityMetrics.calculate_psi(train_scores, prod_scores, bins=10)
        """
        # Convert to numpy arrays
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        # Create bins based on expected distribution
        if isinstance(bins, int):
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # Remove duplicates
        else:
            breakpoints = np.asarray(bins)
        
        # Bin the data
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Calculate proportions
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Add epsilon to avoid log(0)
        expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
        actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
        
        # Calculate PSI
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    
    @staticmethod
    def calculate_csi(
        expected_scores: Union[np.ndarray, pd.Series],
        actual_scores: Union[np.ndarray, pd.Series],
        expected_labels: Union[np.ndarray, pd.Series],
        actual_labels: Union[np.ndarray, pd.Series],
        bins: Union[int, List[float]] = 10,
        epsilon: float = 0.0001,
    ) -> float:
        """
        Calculate Characteristic Stability Index (CSI).
        
        CSI measures the shift in the relationship between features and target.
        
        Interpretation (similar to PSI):
        - CSI < 0.1: No significant change
        - 0.1 <= CSI < 0.2: Moderate change
        - CSI >= 0.2: Significant change (investigate)
        
        Args:
            expected_scores: Reference scores (training/baseline)
            actual_scores: Current scores (production)
            expected_labels: Reference labels (training/baseline)
            actual_labels: Current labels (production)
            bins: Number of bins or explicit bin edges
            epsilon: Small constant to avoid log(0)
        
        Returns:
            CSI value
        
        Example:
            >>> csi = StabilityMetrics.calculate_csi(
            ...     train_scores, prod_scores,
            ...     train_labels, prod_labels
            ... )
        """
        # Convert to numpy arrays
        expected_scores = np.asarray(expected_scores)
        actual_scores = np.asarray(actual_scores)
        expected_labels = np.asarray(expected_labels)
        actual_labels = np.asarray(actual_labels)
        
        # Create bins based on expected distribution
        if isinstance(bins, int):
            breakpoints = np.percentile(expected_scores, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
        else:
            breakpoints = np.asarray(bins)
        
        # Bin the scores
        expected_bins = np.digitize(expected_scores, bins=breakpoints, right=False) - 1
        actual_bins = np.digitize(actual_scores, bins=breakpoints, right=False) - 1
        
        # Calculate event rates per bin
        expected_event_rates = []
        actual_event_rates = []
        
        for i in range(len(breakpoints) - 1):
            # Expected event rate
            exp_mask = expected_bins == i
            if np.sum(exp_mask) > 0:
                exp_rate = np.sum(expected_labels[exp_mask]) / np.sum(exp_mask)
            else:
                exp_rate = epsilon
            expected_event_rates.append(exp_rate)
            
            # Actual event rate
            act_mask = actual_bins == i
            if np.sum(act_mask) > 0:
                act_rate = np.sum(actual_labels[act_mask]) / np.sum(act_mask)
            else:
                act_rate = epsilon
            actual_event_rates.append(act_rate)
        
        expected_event_rates = np.array(expected_event_rates)
        actual_event_rates = np.array(actual_event_rates)
        
        # Add epsilon to avoid log(0)
        expected_event_rates = np.where(expected_event_rates == 0, epsilon, expected_event_rates)
        actual_event_rates = np.where(actual_event_rates == 0, epsilon, actual_event_rates)
        
        # Calculate CSI (similar formula to PSI but for event rates)
        csi_values = (actual_event_rates - expected_event_rates) * np.log(
            actual_event_rates / expected_event_rates
        )
        csi = np.sum(csi_values)
        
        return float(csi)
    
    @staticmethod
    def psi_interpretation(psi: float) -> str:
        """
        Interpret PSI value.
        
        Args:
            psi: PSI value
        
        Returns:
            Interpretation string
        """
        if psi < 0.1:
            return "stable"
        elif psi < 0.2:
            return "warning"
        else:
            return "critical"
    
    @staticmethod
    def calculate_psi_breakdown(
        expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        bins: Union[int, List[float]] = 10,
        epsilon: float = 0.0001,
    ) -> pd.DataFrame:
        """
        Calculate PSI with detailed breakdown by bin.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins or explicit bin edges
            epsilon: Small constant to avoid log(0)
        
        Returns:
            DataFrame with PSI breakdown
        
        Example:
            >>> breakdown = StabilityMetrics.calculate_psi_breakdown(train_scores, prod_scores)
            >>> print(breakdown)
        """
        # Convert to numpy arrays
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        # Create bins
        if isinstance(bins, int):
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)
        else:
            breakpoints = np.asarray(bins)
        
        # Bin the data
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Calculate proportions
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Add epsilon to avoid log(0)
        expected_percents_adj = np.where(expected_percents == 0, epsilon, expected_percents)
        actual_percents_adj = np.where(actual_percents == 0, epsilon, actual_percents)
        
        # Calculate PSI per bin
        psi_values = (actual_percents_adj - expected_percents_adj) * np.log(
            actual_percents_adj / expected_percents_adj
        )
        
        # Create breakdown DataFrame
        breakdown = pd.DataFrame({
            "bin_start": breakpoints[:-1],
            "bin_end": breakpoints[1:],
            "expected_count": expected_counts,
            "actual_count": actual_counts,
            "expected_percent": expected_percents,
            "actual_percent": actual_percents,
            "percent_diff": actual_percents - expected_percents,
            "psi": psi_values,
        })
        
        # Add bin labels
        breakdown["bin_label"] = [
            f"({start:.2f}, {end:.2f}]" 
            for start, end in zip(breakpoints[:-1], breakpoints[1:])
        ]
        
        return breakdown
    
    @staticmethod
    def calculate_feature_stability(
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate PSI for multiple features.
        
        Args:
            expected_df: Reference DataFrame
            actual_df: Current DataFrame
            features: List of features to analyze (None = all numeric)
            bins: Number of bins
        
        Returns:
            DataFrame with PSI per feature
        
        Example:
            >>> stability = StabilityMetrics.calculate_feature_stability(train_df, prod_df)
        """
        if features is None:
            features = expected_df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = []
        
        for feature in features:
            if feature not in expected_df.columns or feature not in actual_df.columns:
                continue
            
            try:
                psi = StabilityMetrics.calculate_psi(
                    expected_df[feature],
                    actual_df[feature],
                    bins=bins
                )
                
                interpretation = StabilityMetrics.psi_interpretation(psi)
                
                results.append({
                    "feature": feature,
                    "psi": psi,
                    "status": interpretation,
                })
            except Exception as e:
                results.append({
                    "feature": feature,
                    "psi": None,
                    "status": f"error: {str(e)}",
                })
        
        return pd.DataFrame(results).sort_values("psi", ascending=False)
    
    @staticmethod
    def kolmogorov_smirnov_test(
        expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            expected: Reference distribution
            actual: Current distribution
        
        Returns:
            Dictionary with KS statistic and p-value
        
        Example:
            >>> ks_result = StabilityMetrics.kolmogorov_smirnov_test(train_scores, prod_scores)
        """
        from scipy.stats import ks_2samp
        
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        statistic, pvalue = ks_2samp(expected, actual)
        
        return {
            "ks_statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": pvalue < 0.05,
        }
