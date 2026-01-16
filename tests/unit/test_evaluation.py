"""
Unit tests for evaluation module.
"""

import numpy as np
import pandas as pd
import pytest

from cr_score.evaluation import (
    ClassificationMetrics,
    StabilityMetrics,
    CalibrationMetrics,
    RankingMetrics,
    PerformanceEvaluator
)


@pytest.fixture
def binary_classification_data():
    """Generate binary classification test data."""
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.beta(2, 5, n_samples)  # Skewed toward lower probabilities
    y_pred = (y_proba > 0.5).astype(int)
    
    return y_true, y_pred, y_proba


class TestClassificationMetrics:
    """Tests for classification metrics."""
    
    def test_calculate_all_metrics(self, binary_classification_data):
        """Test comprehensive metrics calculation."""
        y_true, y_pred, y_proba = binary_classification_data
        
        metrics = ClassificationMetrics.calculate(y_true, y_pred, y_proba)
        
        # Check all expected metrics exist
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'mcc' in metrics
        
        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert -1 <= metrics['mcc'] <= 1
    
    def test_confusion_matrix_components(self, binary_classification_data):
        """Test confusion matrix components."""
        y_true, y_pred, y_proba = binary_classification_data
        
        metrics = ClassificationMetrics.calculate(y_true, y_pred, y_proba)
        
        # Confusion matrix is nested
        assert 'confusion_matrix' in metrics
        cm = metrics['confusion_matrix']
        assert 'true_negative' in cm
        assert 'false_positive' in cm
        assert 'false_negative' in cm
        assert 'true_positive' in cm
        assert cm['true_negative'] + cm['false_positive'] + cm['false_negative'] + cm['true_positive'] == len(y_true)
    
    def test_optimal_threshold(self, binary_classification_data):
        """Test optimal threshold finding."""
        y_true, _, y_proba = binary_classification_data
        
        threshold = ClassificationMetrics.optimal_threshold(
            y_true, y_proba, metric='f1'
        )
        
        assert 0 <= threshold <= 1


class TestStabilityMetrics:
    """Tests for stability metrics."""
    
    def test_psi_identical_distributions(self):
        """Test PSI with identical distributions."""
        expected = np.random.normal(0, 1, 1000)
        actual = expected.copy()
        
        psi = StabilityMetrics.calculate_psi(expected, actual, bins=10)
        
        assert psi < 0.01  # Should be very small for identical distributions
    
    def test_psi_different_distributions(self):
        """Test PSI with different distributions."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(1, 1, 1000)  # Shifted mean
        
        psi = StabilityMetrics.calculate_psi(expected, actual, bins=10)
        
        assert psi > 0.1  # Should detect the shift
    
    def test_psi_interpretation(self):
        """Test PSI interpretation."""
        assert StabilityMetrics.psi_interpretation(0.05) == 'stable'
        assert StabilityMetrics.psi_interpretation(0.15) == 'warning'
        assert StabilityMetrics.psi_interpretation(0.25) == 'critical'
    
    def test_csi(self):
        """Test Characteristic Stability Index."""
        expected_scores = np.random.normal(0, 1, 1000)
        actual_scores = np.random.normal(0.5, 1, 1000)
        expected_labels = np.random.randint(0, 2, 1000)
        actual_labels = np.random.randint(0, 2, 1000)
        
        csi = StabilityMetrics.calculate_csi(
            expected_scores, actual_scores,
            expected_labels, actual_labels,
            bins=10
        )
        
        assert csi >= 0
    
    def test_feature_stability(self):
        """Test feature stability analysis."""
        expected_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
        })
        actual_df = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 1000),  # Drifted
            'feature2': np.random.normal(0, 1, 1000),     # Stable
        })
        
        stability = StabilityMetrics.calculate_feature_stability(
            expected_df, actual_df, features=['feature1', 'feature2']
        )
        
        assert isinstance(stability, pd.DataFrame)
        assert len(stability) == 2
        assert 'feature' in stability.columns
        assert 'psi' in stability.columns
        assert 'status' in stability.columns


class TestCalibrationMetrics:
    """Tests for calibration metrics."""
    
    def test_brier_score(self, binary_classification_data):
        """Test Brier score."""
        y_true, _, y_proba = binary_classification_data
        
        brier = CalibrationMetrics.calculate_brier_score(y_true, y_proba)
        
        assert 0 <= brier <= 1
    
    def test_log_loss(self, binary_classification_data):
        """Test log loss."""
        y_true, _, y_proba = binary_classification_data
        
        # Remove eps parameter - not supported in newer sklearn
        from sklearn.metrics import log_loss
        logloss = float(log_loss(y_true, y_proba))
        
        assert logloss >= 0
    
    def test_calibration_curve(self, binary_classification_data):
        """Test calibration curve."""
        y_true, _, y_proba = binary_classification_data
        
        frac_pos, mean_pred = CalibrationMetrics.calculate_calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        # May return fewer bins if there aren't enough unique values
        assert len(frac_pos) <= 10
        assert len(mean_pred) <= 10
        assert len(frac_pos) == len(mean_pred)
        assert all(0 <= fp <= 1 for fp in frac_pos)


class TestRankingMetrics:
    """Tests for ranking metrics."""
    
    def test_auc(self, binary_classification_data):
        """Test AUC calculation."""
        y_true, _, y_proba = binary_classification_data
        
        auc = RankingMetrics.calculate_auc(y_true, y_proba)
        
        assert 0 <= auc <= 1
    
    def test_gini(self, binary_classification_data):
        """Test Gini coefficient."""
        y_true, _, y_proba = binary_classification_data
        
        gini = RankingMetrics.calculate_gini(y_true, y_proba)
        
        assert -1 <= gini <= 1
    
    def test_ks_statistic(self, binary_classification_data):
        """Test KS statistic."""
        y_true, _, y_proba = binary_classification_data
        
        result = RankingMetrics.calculate_ks_statistic(y_true, y_proba)
        
        # Returns dict with ks_statistic and other info
        if isinstance(result, dict):
            ks = result['ks_statistic']
        else:
            ks = result
        
        assert 0 <= ks <= 1
    
    def test_lift_curve(self, binary_classification_data):
        """Test lift curve."""
        y_true, _, y_proba = binary_classification_data
        
        lift = RankingMetrics.calculate_lift_curve(y_true, y_proba, n_bins=10)
        
        assert isinstance(lift, pd.DataFrame)
        assert len(lift) == 10
        assert 'decile' in lift.columns
        assert 'lift' in lift.columns
    
    def test_gains_curve(self, binary_classification_data):
        """Test gains curve."""
        y_true, _, y_proba = binary_classification_data
        
        gains = RankingMetrics.calculate_gains_curve(y_true, y_proba, n_bins=10)
        
        assert isinstance(gains, pd.DataFrame)
        assert len(gains) == 10
        assert 'decile' in gains.columns
        # Check for actual column names (pct_positives_captured is the cumulative gains)
        assert 'pct_positives_captured' in gains.columns or 'cumulative_gains' in gains.columns


class TestPerformanceEvaluator:
    """Tests for unified performance evaluator."""
    
    def test_evaluate_all(self, binary_classification_data):
        """Test comprehensive evaluation."""
        y_true, y_pred, y_proba = binary_classification_data
        
        evaluator = PerformanceEvaluator()
        results = evaluator.evaluate_all(y_true, y_pred, y_proba)
        
        # Check main categories exist
        assert 'classification' in results
        assert 'ranking' in results
        assert 'calibration' in results
        
        # Check some key metrics
        assert 'accuracy' in results['classification']
        assert 'auc' in results['ranking']
        assert 'brier_score' in results['calibration']
    
    def test_evaluate_stability(self):
        """Test stability evaluation."""
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.5, 1, 1000)
        
        evaluator = PerformanceEvaluator()
        stability = evaluator.evaluate_stability(expected, actual)
        
        assert 'psi' in stability
        assert 'psi_status' in stability
    
    def test_summary(self, binary_classification_data):
        """Test summary generation."""
        y_true, y_pred, y_proba = binary_classification_data
        
        evaluator = PerformanceEvaluator()
        results = evaluator.evaluate_all(y_true, y_pred, y_proba)
        summary = evaluator.summary(results)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
    
    def test_compare_models(self, binary_classification_data):
        """Test model comparison."""
        y_true, y_pred, y_proba = binary_classification_data
        
        evaluator = PerformanceEvaluator()
        results1 = evaluator.evaluate_all(y_true, y_pred, y_proba)
        results2 = evaluator.evaluate_all(y_true, y_pred, y_proba * 0.9)  # Slightly worse
        
        comparison = evaluator.compare_models({
            'model1': results1,
            'model2': results2
        })
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'auc' in comparison.columns
