"""
Unit tests for monitoring module.
"""

import numpy as np
import pandas as pd
import pytest

from cr_score.monitoring import (
    PerformanceMonitor,
    DriftMonitor,
    AlertManager,
    MetricsCollector
)
from cr_score.monitoring.alert_manager import AlertSeverity


@pytest.fixture
def baseline_metrics():
    """Sample baseline metrics."""
    return {
        'auc': 0.85,
        'precision': 0.80,
        'recall': 0.75
    }


@pytest.fixture
def sample_data():
    """Generate sample data for monitoring."""
    np.random.seed(42)
    
    reference = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })
    
    current = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1, 1000),  # Drifted
        'feature2': np.random.normal(0, 1, 1000)      # Stable
    })
    
    return reference, current


class TestPerformanceMonitor:
    """Tests for performance monitoring."""
    
    def test_initialization(self, baseline_metrics):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            alert_thresholds={'auc': 0.05}
        )
        
        assert monitor.baseline_metrics == baseline_metrics
        assert monitor.alert_thresholds['auc'] == 0.05
    
    def test_record_predictions(self, baseline_metrics):
        """Test recording predictions."""
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            alert_thresholds={'auc': 0.05}
        )
        
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        
        metrics = monitor.record_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            metadata={'date': '2026-01-16'}
        )
        
        assert 'auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_check_health(self, baseline_metrics):
        """Test health check."""
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics,
            alert_thresholds={'auc': 0.05}
        )
        
        # Record some predictions
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        
        monitor.record_predictions(y_true, y_pred, y_proba)
        
        health = monitor.check_health()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'warning', 'critical']
    
    def test_get_metrics_summary(self, baseline_metrics):
        """Test metrics summary."""
        monitor = PerformanceMonitor(
            baseline_metrics=baseline_metrics
        )
        
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        
        monitor.record_predictions(y_true, y_pred, y_proba)
        
        summary = monitor.get_metrics_summary()
        
        assert isinstance(summary, pd.DataFrame)


class TestDriftMonitor:
    """Tests for drift monitoring."""
    
    def test_initialization(self, sample_data):
        """Test monitor initialization."""
        reference, _ = sample_data
        
        monitor = DriftMonitor(
            reference_data=reference,
            psi_threshold=0.1,
            ks_threshold=0.05
        )
        
        assert monitor.psi_threshold == 0.1
        assert monitor.ks_threshold == 0.05
    
    def test_detect_drift(self, sample_data):
        """Test drift detection."""
        reference, current = sample_data
        
        monitor = DriftMonitor(
            reference_data=reference,
            psi_threshold=0.1
        )
        
        report = monitor.detect_drift(current)
        
        assert 'overall_status' in report
        assert 'drift_summary' in report
        assert 'feature_results' in report
        
        # Check that feature1 was detected as drifted
        assert 'feature1' in report['feature_results']
    
    def test_get_drift_summary(self, sample_data):
        """Test drift summary."""
        reference, current = sample_data
        
        monitor = DriftMonitor(reference_data=reference)
        report = monitor.detect_drift(current)
        
        summary = report['drift_summary']
        
        assert 'critical' in summary
        assert 'warning' in summary
        assert 'stable' in summary


class TestAlertManager:
    """Tests for alert management."""
    
    def test_create_alert(self):
        """Test alert creation."""
        manager = AlertManager()
        
        alert = manager.create_alert(
            title='Test Alert',
            severity=AlertSeverity.WARNING,
            details={'metric': 'auc', 'value': 0.75},
            source='test'
        )
        
        assert alert['title'] == 'Test Alert'
        assert alert['severity'] == AlertSeverity.WARNING
        assert alert['status'] == 'active'
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        
        manager.create_alert('Alert 1', AlertSeverity.WARNING)
        manager.create_alert('Alert 2', AlertSeverity.CRITICAL)
        
        active = manager.get_active_alerts()
        
        assert len(active) == 2
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        manager = AlertManager()
        
        alert = manager.create_alert('Test Alert', AlertSeverity.WARNING)
        alert_id = alert['id']
        
        manager.resolve_alert(alert_id, resolution='Fixed')
        
        resolved = manager.get_alert(alert_id)
        assert resolved['status'] == 'resolved'
        assert resolved['resolution'] == 'Fixed'
    
    def test_get_alert_summary(self):
        """Test alert summary."""
        manager = AlertManager()
        
        manager.create_alert('Critical Alert', AlertSeverity.CRITICAL)
        manager.create_alert('Warning Alert', AlertSeverity.WARNING)
        
        summary = manager.get_alert_summary()
        
        assert 'total' in summary
        assert 'active' in summary
        assert 'by_severity' in summary
        assert summary['total'] == 2


class TestMetricsCollector:
    """Tests for metrics collection."""
    
    def test_initialization(self):
        """Test collector initialization."""
        collector = MetricsCollector(enable_prometheus=True)
        
        assert collector.enable_prometheus
    
    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment_counter('predictions_total', value=10)
        collector.increment_counter('predictions_total', value=5)
        
        metrics = collector.get_metrics()
        
        assert 'predictions_total' in metrics
        assert metrics['predictions_total']['value'] == 15
    
    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        
        collector.set_gauge('model_auc', value=0.85)
        
        metrics = collector.get_metrics()
        
        assert 'model_auc' in metrics
        assert metrics['model_auc']['value'] == 0.85
    
    def test_record_histogram(self):
        """Test histogram recording."""
        collector = MetricsCollector()
        
        for i in range(100):
            collector.record_histogram('score_value', value=np.random.rand())
        
        metrics = collector.get_metrics()
        
        assert 'score_value' in metrics
        assert 'count' in metrics['score_value']
        assert metrics['score_value']['count'] == 100
    
    def test_get_metric(self):
        """Test getting individual metric."""
        collector = MetricsCollector()
        
        collector.set_gauge('test_metric', value=42)
        
        metric = collector.get_metric('test_metric')
        
        assert metric is not None
        assert metric['value'] == 42
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        
        collector.set_gauge('test_metric', value=42)
        collector.reset_metrics()
        
        metrics = collector.get_metrics()
        
        assert len(metrics) == 0
