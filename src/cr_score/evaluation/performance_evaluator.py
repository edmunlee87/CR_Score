"""
Unified performance evaluator combining all metrics.

Single interface for comprehensive model evaluation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cr_score.evaluation.calibration_metrics import CalibrationMetrics
from cr_score.evaluation.classification_metrics import ClassificationMetrics
from cr_score.evaluation.ranking_metrics import RankingMetrics
from cr_score.evaluation.stability_metrics import StabilityMetrics


class PerformanceEvaluator:
    """
    Unified evaluator for comprehensive model assessment.
    
    Combines classification, ranking, calibration, and stability metrics.
    
    Example:
        >>> evaluator = PerformanceEvaluator()
        >>> results = evaluator.evaluate_all(y_true, y_pred, y_proba)
        >>> print(evaluator.summary(results))
    """
    
    def __init__(self):
        """Initialize performance evaluator."""
        self.classification = ClassificationMetrics()
        self.ranking = RankingMetrics()
        self.calibration = CalibrationMetrics()
        self.stability = StabilityMetrics()
    
    def evaluate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities
            threshold: Classification threshold
        
        Returns:
            Dictionary with all metrics organized by category
        
        Example:
            >>> results = evaluator.evaluate_all(y_true, y_pred, y_proba)
            >>> print(f"AUC: {results['ranking']['auc']:.3f}")
        """
        # Classification metrics
        classification_metrics = self.classification.calculate(
            y_true, y_pred, y_proba, threshold
        )
        
        # Ranking metrics
        ranking_metrics = self.ranking.calculate_all_ranking_metrics(y_true, y_proba)
        
        # Calibration metrics
        calibration_metrics = self.calibration.calculate_all_calibration_metrics(
            y_true, y_proba
        )
        
        return {
            "classification": classification_metrics,
            "ranking": ranking_metrics,
            "calibration": calibration_metrics,
        }
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate only classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities (optional)
            threshold: Classification threshold
        
        Returns:
            Classification metrics
        """
        return self.classification.calculate(y_true, y_pred, y_proba, threshold)
    
    def evaluate_ranking(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate only ranking metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Ranking metrics
        """
        return self.ranking.calculate_all_ranking_metrics(y_true, y_proba)
    
    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate only calibration metrics.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
        
        Returns:
            Calibration metrics
        """
        return self.calibration.calculate_all_calibration_metrics(y_true, y_proba)
    
    def evaluate_stability(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate stability metrics (PSI).
        
        Args:
            expected: Reference/baseline distribution
            actual: Current distribution
            bins: Number of bins
        
        Returns:
            Stability metrics
        """
        psi = self.stability.calculate_psi(expected, actual, bins)
        interpretation = self.stability.psi_interpretation(psi)
        ks_test = self.stability.kolmogorov_smirnov_test(expected, actual)
        
        return {
            "psi": psi,
            "psi_status": interpretation,
            "ks_statistic": ks_test["ks_statistic"],
            "ks_pvalue": ks_test["pvalue"],
            "ks_significant": ks_test["significant"],
        }
    
    def summary(
        self,
        results: Dict[str, Any],
        top_n_metrics: int = 10,
    ) -> pd.DataFrame:
        """
        Create summary DataFrame of key metrics.
        
        Args:
            results: Results from evaluate_all()
            top_n_metrics: Number of top metrics to include
        
        Returns:
            DataFrame with key metrics
        
        Example:
            >>> results = evaluator.evaluate_all(y_true, y_pred, y_proba)
            >>> summary_df = evaluator.summary(results)
            >>> print(summary_df)
        """
        key_metrics = []
        
        # Classification
        if "classification" in results:
            cls = results["classification"]
            key_metrics.extend([
                ("Accuracy", cls.get("accuracy", None)),
                ("Balanced Accuracy", cls.get("balanced_accuracy", None)),
                ("Precision", cls.get("precision", None)),
                ("Recall", cls.get("recall", None)),
                ("F1 Score", cls.get("f1_score", None)),
                ("MCC", cls.get("mcc", None)),
                ("Cohen's Kappa", cls.get("kappa", None)),
            ])
        
        # Ranking
        if "ranking" in results:
            rank = results["ranking"]
            key_metrics.extend([
                ("AUC", rank.get("auc", None)),
                ("Gini", rank.get("gini", None)),
                ("KS Statistic", rank.get("ks_statistic", None)),
                ("Lift @ 10%", rank.get("lift_at_10pct", None)),
                ("Lift @ 20%", rank.get("lift_at_20pct", None)),
                ("H-Measure", rank.get("h_measure", None)),
            ])
        
        # Calibration
        if "calibration" in results:
            calib = results["calibration"]
            key_metrics.extend([
                ("Brier Score", calib.get("brier_score", None)),
                ("Log Loss", calib.get("log_loss", None)),
                ("ECE", calib.get("ece", None)),
                ("MCE", calib.get("mce", None)),
            ])
        
        # Stability
        if "stability" in results:
            stab = results["stability"]
            key_metrics.extend([
                ("PSI", stab.get("psi", None)),
                ("KS Statistic (Drift)", stab.get("ks_statistic", None)),
            ])
        
        # Filter None values and take top N
        key_metrics = [(name, value) for name, value in key_metrics if value is not None]
        key_metrics = key_metrics[:top_n_metrics]
        
        return pd.DataFrame(key_metrics, columns=["Metric", "Value"])
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            models_results: Dict mapping model names to their evaluation results
        
        Returns:
            DataFrame comparing models
        
        Example:
            >>> results_m1 = evaluator.evaluate_all(y_true, y_pred_m1, y_proba_m1)
            >>> results_m2 = evaluator.evaluate_all(y_true, y_pred_m2, y_proba_m2)
            >>> comparison = evaluator.compare_models({
            ...     'Model 1': results_m1,
            ...     'Model 2': results_m2
            ... })
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {"model": model_name}
            
            # Extract key metrics
            if "classification" in results:
                row.update({
                    "accuracy": results["classification"].get("accuracy"),
                    "precision": results["classification"].get("precision"),
                    "recall": results["classification"].get("recall"),
                    "f1": results["classification"].get("f1_score"),
                    "mcc": results["classification"].get("mcc"),
                })
            
            if "ranking" in results:
                row.update({
                    "auc": results["ranking"].get("auc"),
                    "gini": results["ranking"].get("gini"),
                    "ks": results["ranking"].get("ks_statistic"),
                })
            
            if "calibration" in results:
                row.update({
                    "brier": results["calibration"].get("brier_score"),
                    "log_loss": results["calibration"].get("log_loss"),
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def interpret_results(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Provide interpretations of metric values.
        
        Args:
            results: Results from evaluate_all()
        
        Returns:
            Dictionary with interpretations
        
        Example:
            >>> interpretations = evaluator.interpret_results(results)
            >>> print(interpretations['auc'])
        """
        interpretations = {}
        
        # AUC interpretation
        if "ranking" in results:
            auc_val = results["ranking"].get("auc", 0)
            if auc_val >= 0.9:
                interpretations["auc"] = "Excellent discrimination"
            elif auc_val >= 0.8:
                interpretations["auc"] = "Good discrimination"
            elif auc_val >= 0.7:
                interpretations["auc"] = "Acceptable discrimination"
            elif auc_val >= 0.6:
                interpretations["auc"] = "Poor discrimination"
            else:
                interpretations["auc"] = "Very poor discrimination"
            
            # Gini interpretation
            gini_val = results["ranking"].get("gini", 0)
            if gini_val >= 0.6:
                interpretations["gini"] = "Excellent model power"
            elif gini_val >= 0.4:
                interpretations["gini"] = "Good model power"
            elif gini_val >= 0.2:
                interpretations["gini"] = "Weak model power"
            else:
                interpretations["gini"] = "Very weak model power"
            
            # KS interpretation
            ks_val = results["ranking"].get("ks_statistic", 0)
            if ks_val >= 0.4:
                interpretations["ks"] = "Strong separation"
            elif ks_val >= 0.3:
                interpretations["ks"] = "Good separation"
            elif ks_val >= 0.2:
                interpretations["ks"] = "Moderate separation"
            else:
                interpretations["ks"] = "Weak separation"
        
        # Brier score interpretation
        if "calibration" in results:
            brier_val = results["calibration"].get("brier_score", 1)
            if brier_val <= 0.1:
                interpretations["brier"] = "Excellent calibration"
            elif brier_val <= 0.2:
                interpretations["brier"] = "Good calibration"
            else:
                interpretations["brier"] = "Poor calibration"
        
        # PSI interpretation
        if "stability" in results:
            psi_val = results["stability"].get("psi", 0)
            interpretations["psi"] = results["stability"].get("psi_status", "unknown")
        
        return interpretations
