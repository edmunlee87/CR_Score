"""
Ranking metrics for credit scorecard evaluation.

Gini, KS, Lift, Gains, and other ranking-based metrics.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


class RankingMetrics:
    """
    Ranking-based evaluation metrics for scorecards.
    
    Focuses on how well the model ranks good vs bad accounts.
    
    Example:
        >>> metrics = RankingMetrics()
        >>> gini = metrics.calculate_gini(y_true, y_proba)
        >>> print(f"Gini: {gini:.3f}")
    """
    
    @staticmethod
    def calculate_auc(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate Area Under ROC Curve (AUC).
        
        Range: [0.5, 1.0]
        - 0.5: Random classifier
        - 1.0: Perfect classifier
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            AUC score
        
        Example:
            >>> auc_score = RankingMetrics.calculate_auc(y_true, y_proba)
        """
        return float(roc_auc_score(y_true, y_proba))
    
    @staticmethod
    def calculate_gini(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate Gini coefficient.
        
        Gini = 2 * AUC - 1
        
        Range: [0, 1]
        - 0: Random classifier
        - 1: Perfect classifier
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Gini coefficient
        
        Example:
            >>> gini = RankingMetrics.calculate_gini(y_true, y_proba)
        """
        auc_score = roc_auc_score(y_true, y_proba)
        return float(2 * auc_score - 1)
    
    @staticmethod
    def calculate_ks_statistic(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate Kolmogorov-Smirnov (KS) statistic.
        
        KS measures maximum separation between good and bad distributions.
        
        Range: [0, 1]
        - 0: No separation
        - 1: Perfect separation
        - Typical good scorecard: KS > 0.4
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Dictionary with KS statistic and threshold
        
        Example:
            >>> ks_result = RankingMetrics.calculate_ks_statistic(y_true, y_proba)
            >>> print(f"KS: {ks_result['ks_statistic']:.3f}")
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # KS is maximum distance between TPR and FPR
        ks_statistic = np.max(tpr - fpr)
        ks_idx = np.argmax(tpr - fpr)
        ks_threshold = float(thresholds[ks_idx]) if ks_idx < len(thresholds) else 0.5
        
        return {
            "ks_statistic": float(ks_statistic),
            "ks_threshold": ks_threshold,
            "ks_tpr": float(tpr[ks_idx]),
            "ks_fpr": float(fpr[ks_idx]),
        }
    
    @staticmethod
    def calculate_lift_at_k(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        k: float = 0.1,
    ) -> float:
        """
        Calculate lift at top k% of predictions.
        
        Lift measures how much better the model is than random targeting.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            k: Fraction of top predictions to consider (e.g., 0.1 for top 10%)
        
        Returns:
            Lift value
        
        Example:
            >>> lift = RankingMetrics.calculate_lift_at_k(y_true, y_proba, k=0.1)
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Take top k%
        n_top = int(len(y_true) * k)
        if n_top == 0:
            n_top = 1
        
        # Calculate lift
        baseline_conversion = np.mean(y_true)
        top_k_conversion = np.mean(y_true_sorted[:n_top])
        
        lift = top_k_conversion / baseline_conversion if baseline_conversion > 0 else 0
        
        return float(lift)
    
    @staticmethod
    def calculate_gains_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate cumulative gains curve data.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins/deciles
        
        Returns:
            DataFrame with gains curve data
        
        Example:
            >>> gains = RankingMetrics.calculate_gains_curve(y_true, y_proba, n_bins=10)
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_proba_sorted = y_proba[sorted_indices]
        
        # Calculate cumulative metrics
        n = len(y_true)
        n_pos = np.sum(y_true)
        bin_size = n // n_bins
        
        results = []
        
        for i in range(1, n_bins + 1):
            idx = min(i * bin_size, n)
            
            # Cumulative counts
            cum_count = idx
            cum_pos = np.sum(y_true_sorted[:idx])
            
            # Percentages
            pct_population = (cum_count / n) * 100
            pct_positives_captured = (cum_pos / n_pos * 100) if n_pos > 0 else 0
            
            # Lift
            baseline_pct = (cum_count / n) * 100
            lift = pct_positives_captured / baseline_pct if baseline_pct > 0 else 0
            
            results.append({
                "decile": i,
                "threshold": float(y_proba_sorted[idx - 1]) if idx > 0 else 1.0,
                "cumulative_count": cum_count,
                "cumulative_positives": int(cum_pos),
                "pct_population": pct_population,
                "pct_positives_captured": pct_positives_captured,
                "lift": lift,
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_lift_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calculate lift curve data (non-cumulative).
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins/deciles
        
        Returns:
            DataFrame with lift curve data
        
        Example:
            >>> lift = RankingMetrics.calculate_lift_curve(y_true, y_proba, n_bins=10)
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        y_proba_sorted = y_proba[sorted_indices]
        
        # Calculate per-bin metrics
        n = len(y_true)
        baseline_rate = np.mean(y_true)
        bin_size = n // n_bins
        
        results = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
            
            bin_y_true = y_true_sorted[start_idx:end_idx]
            bin_y_proba = y_proba_sorted[start_idx:end_idx]
            
            bin_rate = np.mean(bin_y_true)
            lift = bin_rate / baseline_rate if baseline_rate > 0 else 0
            
            results.append({
                "decile": i + 1,
                "min_score": float(np.min(bin_y_proba)),
                "max_score": float(np.max(bin_y_proba)),
                "count": len(bin_y_true),
                "positives": int(np.sum(bin_y_true)),
                "positive_rate": bin_rate,
                "lift": lift,
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_cap_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Cumulative Accuracy Profile (CAP) curve.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Tuple of (cumulative_fraction_of_population, cumulative_fraction_of_positives)
        
        Example:
            >>> x, y = RankingMetrics.calculate_cap_curve(y_true, y_proba)
        """
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate cumulative fractions
        n = len(y_true)
        n_pos = np.sum(y_true)
        
        cum_pos = np.cumsum(y_true_sorted)
        
        # Fractions
        frac_population = np.arange(1, n + 1) / n
        frac_positives = cum_pos / n_pos if n_pos > 0 else cum_pos
        
        # Add (0, 0) point
        frac_population = np.concatenate([[0], frac_population])
        frac_positives = np.concatenate([[0], frac_positives])
        
        return frac_population, frac_positives
    
    @staticmethod
    def calculate_accuracy_ratio(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate Accuracy Ratio (AR) from CAP curve.
        
        AR is the ratio of the area between model CAP and random CAP
        to the area between perfect CAP and random CAP.
        
        AR is equivalent to Gini coefficient.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Accuracy ratio
        
        Example:
            >>> ar = RankingMetrics.calculate_accuracy_ratio(y_true, y_proba)
        """
        # This is equivalent to Gini
        return RankingMetrics.calculate_gini(y_true, y_proba)
    
    @staticmethod
    def calculate_h_measure(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cost_ratio: float = 1.0,
    ) -> float:
        """
        Calculate H-measure (coherent alternative to AUC).
        
        H-measure is a coherent performance measure that addresses
        some limitations of AUC.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            cost_ratio: Ratio of misclassification costs (FP cost / FN cost)
        
        Returns:
            H-measure
        
        Example:
            >>> h = RankingMetrics.calculate_h_measure(y_true, y_proba)
        """
        from sklearn.metrics import confusion_matrix as cm
        
        # Find optimal threshold based on cost ratio
        thresholds = np.unique(y_proba)
        min_loss = float('inf')
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = cm(y_true, y_pred).ravel()
            
            # Weighted loss
            loss = cost_ratio * fp + fn
            
            if loss < min_loss:
                min_loss = loss
        
        # Baseline loss (always predict most frequent class)
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        baseline_loss = min(cost_ratio * n_pos, n_neg)
        
        # H-measure
        if baseline_loss > 0:
            h_measure = 1 - (min_loss / baseline_loss)
        else:
            h_measure = 0
        
        return float(max(0, h_measure))  # Ensure non-negative
    
    @staticmethod
    def calculate_all_ranking_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate all ranking metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Dictionary with all ranking metrics
        
        Example:
            >>> metrics = RankingMetrics.calculate_all_ranking_metrics(y_true, y_proba)
        """
        auc_score = RankingMetrics.calculate_auc(y_true, y_proba)
        gini = RankingMetrics.calculate_gini(y_true, y_proba)
        ks_result = RankingMetrics.calculate_ks_statistic(y_true, y_proba)
        lift_10 = RankingMetrics.calculate_lift_at_k(y_true, y_proba, k=0.1)
        lift_20 = RankingMetrics.calculate_lift_at_k(y_true, y_proba, k=0.2)
        h_measure = RankingMetrics.calculate_h_measure(y_true, y_proba)
        
        return {
            "auc": auc_score,
            "gini": gini,
            "ks_statistic": ks_result["ks_statistic"],
            "ks_threshold": ks_result["ks_threshold"],
            "lift_at_10pct": lift_10,
            "lift_at_20pct": lift_20,
            "h_measure": h_measure,
            "accuracy_ratio": gini,  # Same as Gini
        }
