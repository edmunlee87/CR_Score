"""
Classification metrics for binary classification models.

Model-agnostic metrics that work with any classifier.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


class ClassificationMetrics:
    """
    Comprehensive classification metrics for binary classification.
    
    Provides model-agnostic metrics calculation for any binary classifier.
    
    Example:
        >>> metrics = ClassificationMetrics()
        >>> results = metrics.calculate(y_true, y_pred, y_proba)
        >>> print(f"Accuracy: {results['accuracy']:.3f}")
        >>> print(f"MCC: {results['mcc']:.3f}")
    """
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_proba: Predicted probabilities (optional)
            threshold: Classification threshold for converting probas to labels
            beta: Beta parameter for F-beta score
        
        Returns:
            Dictionary with all classification metrics
        
        Example:
            >>> metrics_dict = ClassificationMetrics.calculate(y_true, y_pred, y_proba)
        """
        # If probabilities provided, generate predictions
        if y_proba is not None and len(y_pred) != len(y_true):
            y_pred = (y_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F-scores
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        
        # Specificity and Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        
        # True/False rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Positive/Negative Predictive Values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Markedness and Informedness
        markedness = ppv + npv - 1
        informedness = tpr + tnr - 1  # Also known as Youden's J statistic
        
        # Diagnostic Odds Ratio
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        
        # Likelihood Ratios
        lr_positive = tpr / fpr if fpr > 0 else float('inf')
        lr_negative = fnr / tnr if tnr > 0 else float('inf')
        
        # Prevalence and Detection Rate
        prevalence = (tp + fn) / (tp + tn + fp + fn)
        detection_rate = tp / (tp + tn + fp + fn)
        detection_prevalence = (tp + fp) / (tp + tn + fp + fn)
        
        return {
            # Basic metrics
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            
            # Precision, Recall, F-scores
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "fbeta_score": float(fbeta),
            
            # Correlation metrics
            "mcc": float(mcc),  # Matthews Correlation Coefficient
            "kappa": float(kappa),  # Cohen's Kappa
            
            # Rates
            "tpr": float(tpr),  # True Positive Rate
            "fpr": float(fpr),  # False Positive Rate
            "tnr": float(tnr),  # True Negative Rate
            "fnr": float(fnr),  # False Negative Rate
            
            # Predictive Values
            "ppv": float(ppv),  # Positive Predictive Value
            "npv": float(npv),  # Negative Predictive Value
            
            # Advanced metrics
            "markedness": float(markedness),
            "informedness": float(informedness),  # Youden's J
            "dor": float(dor) if not np.isinf(dor) else None,  # Diagnostic Odds Ratio
            "lr_positive": float(lr_positive) if not np.isinf(lr_positive) else None,
            "lr_negative": float(lr_negative) if not np.isinf(lr_negative) else None,
            
            # Prevalence metrics
            "prevalence": float(prevalence),
            "detection_rate": float(detection_rate),
            "detection_prevalence": float(detection_prevalence),
            
            # Confusion Matrix
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            
            # Additional info
            "threshold": float(threshold),
            "n_samples": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(len(y_true) - np.sum(y_true)),
        }
    
    @staticmethod
    def calculate_per_class(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            DataFrame with per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        return pd.DataFrame({
            "class": [0, 1],
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        })
    
    @staticmethod
    def optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "youden",
    ) -> float:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Optimization metric ('youden', 'f1', 'precision', 'recall')
        
        Returns:
            Optimal threshold value
        
        Example:
            >>> threshold = ClassificationMetrics.optimal_threshold(y_true, y_proba, metric='youden')
        """
        from sklearn.metrics import roc_curve
        
        if metric == "youden":
            # Youden's J statistic (sensitivity + specificity - 1)
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            return float(thresholds[optimal_idx])
        
        else:
            # Grid search for best metric
            thresholds = np.linspace(0.01, 0.99, 99)
            best_score = -np.inf
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                
                if metric == "f1":
                    score = f1_score(y_true, y_pred, zero_division=0)
                elif metric == "precision":
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == "recall":
                    score = recall_score(y_true, y_pred, zero_division=0)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            return float(best_threshold)
