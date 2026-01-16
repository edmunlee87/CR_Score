"""
Prediction monitoring for production scorecards.

Monitors prediction distributions, outliers, and anomalies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class PredictionMonitor:
    """
    Monitor predictions in production.
    
    Tracks prediction distributions, detects anomalies, and monitors score ranges.
    
    Example:
        >>> monitor = PredictionMonitor(expected_min=300, expected_max=850)
        >>> monitor.record_predictions(scores, probabilities)
        >>> anomalies = monitor.detect_anomalies()
    """
    
    def __init__(
        self,
        expected_score_range: Optional[tuple] = (300, 850),
        expected_proba_range: tuple = (0.0, 1.0),
        storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize prediction monitor.
        
        Args:
            expected_score_range: Expected score range (min, max)
            expected_proba_range: Expected probability range
            storage_path: Path to store monitoring data
        """
        self.expected_score_range = expected_score_range
        self.expected_proba_range = expected_proba_range
        self.storage_path = Path(storage_path) if storage_path else Path('./monitoring_data')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_audit_logger()
        self.prediction_history: List[Dict[str, Any]] = []
    
    def record_predictions(
        self,
        scores: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record batch of predictions.
        
        Args:
            scores: Credit scores
            probabilities: Default probabilities
            metadata: Additional metadata
        
        Returns:
            Prediction statistics
        """
        timestamp = datetime.now().isoformat()
        
        stats = {
            'timestamp': timestamp,
            'n_predictions': len(scores),
            'score_mean': float(np.mean(scores)),
            'score_median': float(np.median(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(np.min(scores)),
            'score_max': float(np.max(scores)),
            'score_q25': float(np.quantile(scores, 0.25)),
            'score_q75': float(np.quantile(scores, 0.75)),
        }
        
        # Out of range scores
        if self.expected_score_range:
            out_of_range = np.sum(
                (scores < self.expected_score_range[0]) |
                (scores > self.expected_score_range[1])
            )
            stats['scores_out_of_range'] = int(out_of_range)
            stats['pct_out_of_range'] = float(out_of_range / len(scores) * 100)
        
        # Probability statistics
        if probabilities is not None:
            stats['proba_mean'] = float(np.mean(probabilities))
            stats['proba_median'] = float(np.median(probabilities))
            stats['proba_std'] = float(np.std(probabilities))
        
        # Add metadata
        if metadata:
            stats.update(metadata)
        
        # Save
        self.prediction_history.append(stats)
        self._save_stats(stats)
        
        self.logger.info(
            "Recorded predictions",
            n_predictions=stats['n_predictions'],
            score_mean=stats['score_mean'],
        )
        
        return stats
    
    def _save_stats(self, stats: Dict[str, Any]) -> None:
        """Save statistics to disk."""
        stats_file = self.storage_path / 'prediction_history.jsonl'
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
    
    def detect_anomalies(
        self,
        scores: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Detect anomalous predictions.
        
        Args:
            scores: Credit scores
            method: Detection method ('iqr', 'zscore')
            threshold: Anomaly threshold
        
        Returns:
            Anomaly detection results
        """
        anomalies = []
        
        if method == "iqr":
            q25, q75 = np.quantile(scores, [0.25, 0.75])
            iqr = q75 - q25
            lower_bound = q25 - threshold * iqr
            upper_bound = q75 + threshold * iqr
            anomaly_mask = (scores < lower_bound) | (scores > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs((scores - np.mean(scores)) / np.std(scores))
            anomaly_mask = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        anomaly_indices = np.where(anomaly_mask)[0]
        
        result = {
            'method': method,
            'threshold': threshold,
            'n_anomalies': int(np.sum(anomaly_mask)),
            'pct_anomalies': float(np.sum(anomaly_mask) / len(scores) * 100),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': scores[anomaly_mask].tolist(),
        }
        
        if result['n_anomalies'] > 0:
            self.logger.warning(
                "Anomalies detected",
                n_anomalies=result['n_anomalies'],
                pct=result['pct_anomalies'],
            )
        
        return result
    
    def get_prediction_summary(self) -> pd.DataFrame:
        """Get summary of prediction history."""
        if not self.prediction_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.prediction_history)
    
    def plot_prediction_distribution(
        self,
        scores: np.ndarray,
        title: str = "Score Distribution",
    ) -> None:
        """Plot prediction distribution."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.0f}')
        plt.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.0f}')
        
        if self.expected_score_range:
            plt.axvline(self.expected_score_range[0], color='orange', linestyle=':', label='Expected Range')
            plt.axvline(self.expected_score_range[1], color='orange', linestyle=':')
        
        plt.xlabel('Credit Score')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
