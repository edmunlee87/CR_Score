"""
Unit tests for reporting module.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from cr_score.reporting import ReportExporter


@pytest.fixture
def sample_metrics():
    """Generate sample metrics for testing."""
    return {
        'classification': {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'mcc': 0.65
        },
        'ranking': {
            'auc': 0.88,
            'gini': 0.76,
            'ks_statistic': 0.45
        },
        'calibration': {
            'brier_score': 0.12,
            'log_loss': 0.35,
            'ece': 0.08
        },
        'stability': {
            'psi': 0.08,
            'psi_status': 'stable'
        },
        'interpretations': {
            'auc': 'Good discrimination',
            'gini': 'Strong ranking power'
        }
    }


@pytest.fixture
def sample_model():
    """Create a simple mock model."""
    class MockModel:
        def get_feature_importance(self):
            return pd.DataFrame({
                'feature': ['age', 'income', 'credit_score'],
                'importance': [0.5, 0.3, 0.2]
            })
        
        def get_lift_curve(self, y_true, y_proba, n_bins=10):
            return pd.DataFrame({
                'decile': range(1, n_bins+1),
                'lift': np.random.uniform(1, 3, n_bins)
            })
        
        def get_gains_curve(self, y_true, y_proba, n_bins=10):
            return pd.DataFrame({
                'decile': range(1, n_bins+1),
                'cumulative_gains': np.linspace(0, 1, n_bins)
            })
    
    return MockModel()


class TestReportExporter:
    """Tests for report exporter."""
    
    def test_export_metrics_json(self, sample_metrics):
        """Test JSON export."""
        exporter = ReportExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'metrics.json'
            exporter.export_metrics_json(sample_metrics, filepath, pretty=True)
            
            assert filepath.exists()
            
            # Verify content
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'metrics' in data
            assert 'timestamp' in data
            assert data['metrics']['classification']['accuracy'] == 0.85
    
    def test_export_metrics_csv(self, sample_metrics):
        """Test CSV export."""
        exporter = ReportExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = exporter.export_metrics_csv(sample_metrics, tmpdir)
            
            assert len(files) > 0
            assert all(f.exists() for f in files)
            
            # Check classification metrics file
            clf_file = Path(tmpdir) / 'classification_metrics.csv'
            assert clf_file.exists()
            
            df = pd.read_csv(clf_file)
            assert 'accuracy' in df.columns
    
    def test_export_to_excel(self, sample_metrics):
        """Test Excel export."""
        exporter = ReportExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'report.xlsx'
            exporter.export_to_excel(
                metrics=sample_metrics,
                filepath=filepath
            )
            
            assert filepath.exists()
            
            # Verify content
            df = pd.read_excel(filepath, sheet_name='Summary')
            assert len(df) > 0
    
    def test_export_to_markdown(self, sample_metrics):
        """Test Markdown export."""
        exporter = ReportExporter()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'report.md'
            exporter.export_to_markdown(sample_metrics, filepath=filepath)
            
            assert filepath.exists()
            
            # Verify content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '# Scorecard Report' in content
            assert 'Classification Metrics' in content
    
    def test_export_comprehensive_report(self, sample_metrics, sample_model):
        """Test comprehensive report export."""
        exporter = ReportExporter()
        
        # Create mock data
        X_test = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
        y_test = pd.Series(np.random.randint(0, 2, 100))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = exporter.export_comprehensive_report(
                model=sample_model,
                metrics=sample_metrics,
                X_test=X_test,
                y_test=y_test,
                output_dir=tmpdir,
                formats=['json', 'csv', 'markdown'],
                include_curves=True
            )
            
            assert 'json' in files
            assert 'csv' in files
            assert 'markdown' in files
            assert all(len(paths) > 0 for paths in files.values())
    
    def test_convert_to_serializable(self):
        """Test numpy/pandas to JSON serialization."""
        exporter = ReportExporter()
        
        # Test numpy types
        data = {
            'np_int': np.int64(42),
            'np_float': np.float64(3.14),
            'np_array': np.array([1, 2, 3]),
            'pd_series': pd.Series([4, 5, 6]),
            'pd_df': pd.DataFrame({'a': [7, 8]})
        }
        
        serializable = exporter._convert_to_serializable(data)
        
        # Should be JSON serializable now
        json_str = json.dumps(serializable)
        assert json_str is not None
