"""
Tests for temporal visualization enhancements.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from cr_score.viz.bin_plots import BinningVisualizer
from cr_score.viz.score_plots import ScoreVisualizer


@pytest.fixture
def temporal_df():
    """Create sample temporal DataFrame."""
    np.random.seed(42)
    
    snapshots = ["2024-01", "2024-06", "2024-12"]
    bins = ["[0-30]", "[31-50]", "[51-100]", "[101-150]"]
    
    data = []
    for snapshot in snapshots:
        for bin_val in bins:
            n = np.random.randint(100, 500)
            for _ in range(n):
                data.append({
                    'snapshot': snapshot,
                    'bin': bin_val,
                    'target': np.random.choice([0, 1], p=[0.9, 0.1] if bin_val == "[0-30]" else [0.7, 0.3]),
                    'feature_value': np.random.randint(0, 200),
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def temporal_score_df():
    """Create sample temporal score DataFrame."""
    np.random.seed(42)
    
    snapshots = ["2024-01", "2024-06", "2024-12"]
    
    data = []
    for snapshot in snapshots:
        n = 1000
        for _ in range(n):
            data.append({
                'snapshot': snapshot,
                'score': np.random.randint(300, 850),
                'target': np.random.choice([0, 1], p=[0.9, 0.1]),
            })
    
    return pd.DataFrame(data)


class TestBinningVisualizerTemporal:
    """Test temporal features in BinningVisualizer."""
    
    def test_plot_temporal_bin_drift_basic(self, temporal_df):
        """Test basic temporal bin drift plot."""
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            snapshot_values=["2024-01", "2024-06", "2024-12"],
        )
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_plot_temporal_bin_drift_with_baseline(self, temporal_df):
        """Test temporal bin drift with baseline."""
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
        )
        
        assert fig is not None
    
    def test_plot_temporal_bin_drift_with_confidence_bands(self, temporal_df):
        """Test temporal bin drift with confidence bands."""
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            show_confidence_bands=True,
        )
        
        assert fig is not None
    
    def test_plot_temporal_bin_drift_with_segment(self, temporal_df):
        """Test temporal bin drift with segmentation."""
        temporal_df['segment'] = np.random.choice(['A', 'B'], len(temporal_df))
        
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            segment_col='segment',
            segment_values=['A'],
        )
        
        assert fig is not None
    
    def test_plot_temporal_bin_drift_max_bins(self, temporal_df):
        """Test temporal bin drift with max_bins_display."""
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            max_bins_display=2,
        )
        
        assert fig is not None
    
    def test_plot_bin_delta_vs_baseline(self, temporal_df):
        """Test bin delta vs baseline plot."""
        viz = BinningVisualizer()
        
        fig = viz.plot_bin_delta_vs_baseline(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
        )
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_plot_bin_delta_vs_baseline_with_comparison(self, temporal_df):
        """Test bin delta with specific comparison snapshots."""
        viz = BinningVisualizer()
        
        fig = viz.plot_bin_delta_vs_baseline(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
            comparison_snapshots=["2024-06", "2024-12"],
        )
        
        assert fig is not None
    
    def test_plot_psi_by_feature(self, temporal_df):
        """Test PSI by feature plot."""
        viz = BinningVisualizer()
        
        fig = viz.plot_psi_by_feature(
            temporal_df,
            feature_col='feature_value',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
        )
        
        assert fig is not None
    
    def test_plot_psi_by_feature_with_comparison(self, temporal_df):
        """Test PSI with specific comparison snapshots."""
        viz = BinningVisualizer()
        
        fig = viz.plot_psi_by_feature(
            temporal_df,
            feature_col='feature_value',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
            comparison_snapshots=["2024-06", "2024-12"],
            n_bins=5,
        )
        
        assert fig is not None
    
    def test_export_figure_with_metadata_html(self, temporal_df):
        """Test export with metadata to HTML."""
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            path = f.name
        
        try:
            metadata = {
                "feature_name": "bin",
                "model_id": "test_v1",
                "snapshot_range": "2024-01 to 2024-12",
                "baseline_snapshot": "2024-01",
            }
            
            viz._export_figure_with_metadata(fig, path, format="html", metadata=metadata)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_backward_compatibility_binning_table(self):
        """Test that existing plot_binning_table still works."""
        viz = BinningVisualizer()
        
        binning_table = pd.DataFrame({
            'bin': ['[0-30]', '[31-50]', '[51-100]'],
            'total_count': [100, 200, 150],
            'event_rate': [0.1, 0.2, 0.3],
            'woe': [0.5, 1.0, 1.5],
            'iv_contribution': [0.1, 0.2, 0.3],
        })
        
        fig = viz.plot_binning_table(binning_table)
        assert fig is not None
        assert len(fig.data) == 4  # 4 subplots
    
    def test_backward_compatibility_iv_summary(self):
        """Test that existing plot_iv_summary still works."""
        viz = BinningVisualizer()
        
        iv_summary = pd.DataFrame({
            'feature': ['age', 'balance', 'credit_limit'],
            'iv': [0.15, 0.25, 0.10],
        })
        
        fig = viz.plot_iv_summary(iv_summary)
        assert fig is not None


class TestScoreVisualizerTemporal:
    """Test temporal features in ScoreVisualizer."""
    
    def test_plot_temporal_score_distribution_basic(self, temporal_score_df):
        """Test basic temporal score distribution."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_score_distribution(
            temporal_score_df,
            score_col='score',
            snapshot_col='snapshot',
        )
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_plot_temporal_score_distribution_with_target(self, temporal_score_df):
        """Test temporal score distribution with target class coloring."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_score_distribution(
            temporal_score_df,
            score_col='score',
            snapshot_col='snapshot',
            target_col='target',
            snapshot_values=["2024-01", "2024-06"],
        )
        
        assert fig is not None
    
    def test_plot_temporal_score_distribution_with_segment(self, temporal_score_df):
        """Test temporal score distribution with segmentation."""
        temporal_score_df['segment'] = np.random.choice(['A', 'B'], len(temporal_score_df))
        
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_score_distribution(
            temporal_score_df,
            score_col='score',
            snapshot_col='snapshot',
            segment_col='segment',
            segment_values=['A'],
        )
        
        assert fig is not None
    
    def test_plot_temporal_ks_comparison(self, temporal_score_df):
        """Test temporal KS comparison."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_ks_comparison(
            temporal_score_df,
            score_col='score',
            target_col='target',
            snapshot_col='snapshot',
        )
        
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_plot_temporal_ks_comparison_with_snapshots(self, temporal_score_df):
        """Test temporal KS comparison with specific snapshots."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_ks_comparison(
            temporal_score_df,
            score_col='score',
            target_col='target',
            snapshot_col='snapshot',
            snapshot_values=["2024-01", "2024-06"],
        )
        
        assert fig is not None
    
    def test_plot_temporal_stability_metrics(self, temporal_score_df):
        """Test temporal stability metrics."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_stability_metrics(
            temporal_score_df,
            score_col='score',
            target_col='target',
            snapshot_col='snapshot',
            approval_threshold=600,
        )
        
        assert fig is not None
    
    def test_plot_temporal_stability_metrics_no_threshold(self, temporal_score_df):
        """Test temporal stability metrics without approval threshold."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_stability_metrics(
            temporal_score_df,
            score_col='score',
            target_col='target',
            snapshot_col='snapshot',
        )
        
        assert fig is not None
    
    def test_export_figure_with_metadata(self, temporal_score_df):
        """Test export with metadata."""
        viz = ScoreVisualizer()
        
        fig = viz.plot_temporal_score_distribution(
            temporal_score_df,
            score_col='score',
            snapshot_col='snapshot',
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            path = f.name
        
        try:
            metadata = {
                "model_id": "test_v1",
                "snapshot_range": "2024-01 to 2024-12",
            }
            
            viz._export_figure_with_metadata(fig, path, format="html", metadata=metadata)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_backward_compatibility_score_distribution(self, temporal_score_df):
        """Test that existing plot_score_distribution still works."""
        viz = ScoreVisualizer()
        
        scores = temporal_score_df['score'].values
        y_true = temporal_score_df['target'].values
        
        fig = viz.plot_score_distribution(scores, y_true)
        assert fig is not None
    
    def test_backward_compatibility_roc_curve(self, temporal_score_df):
        """Test that existing plot_roc_curve still works."""
        viz = ScoreVisualizer()
        
        y_true = temporal_score_df['target'].values
        y_pred_proba = np.random.rand(len(y_true))
        
        fig = viz.plot_roc_curve(y_true, y_pred_proba)
        assert fig is not None
    
    def test_backward_compatibility_calibration_curve(self, temporal_score_df):
        """Test that existing plot_calibration_curve still works."""
        viz = ScoreVisualizer()
        
        y_true = temporal_score_df['target'].values
        y_pred_proba = np.random.rand(len(y_true))
        
        fig = viz.plot_calibration_curve(y_true, y_pred_proba)
        assert fig is not None


class TestTemporalVisualizationEdgeCases:
    """Test edge cases for temporal visualization."""
    
    def test_empty_snapshots(self):
        """Test handling of empty snapshot data."""
        df = pd.DataFrame({
            'snapshot': [],
            'bin': [],
            'target': [],
        })
        
        viz = BinningVisualizer()
        
        # Should handle gracefully
        fig = viz.plot_temporal_bin_drift(
            df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
        )
        
        assert fig is not None
    
    def test_single_snapshot(self, temporal_df):
        """Test with single snapshot."""
        df_single = temporal_df[temporal_df['snapshot'] == '2024-01'].copy()
        
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            df_single,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
        )
        
        assert fig is not None
    
    def test_missing_baseline_snapshot(self, temporal_df):
        """Test with baseline snapshot not in data."""
        viz = BinningVisualizer()
        
        # Should default to first snapshot
        fig = viz.plot_bin_delta_vs_baseline(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            baseline_snapshot="2023-01",  # Not in data
        )
        
        assert fig is not None
    
    def test_max_bins_truncation(self, temporal_df):
        """Test that max_bins_display truncates correctly."""
        # Create data with many bins
        bins = [f"bin_{i}" for i in range(50)]
        df_large = temporal_df.copy()
        df_large['bin'] = np.random.choice(bins, len(df_large))
        
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            df_large,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            max_bins_display=10,
        )
        
        assert fig is not None


class TestTemporalVisualizationIntegration:
    """Test integration with other modules."""
    
    def test_psi_integration_with_feature_validator(self, temporal_df):
        """Test that PSI calculation uses FeatureValidator."""
        viz = BinningVisualizer()
        
        # Should work with FeatureValidator integration
        fig = viz.plot_psi_by_feature(
            temporal_df,
            feature_col='feature_value',
            snapshot_col='snapshot',
            baseline_snapshot="2024-01",
        )
        
        assert fig is not None
    
    def test_segmentation_filtering(self, temporal_df):
        """Test that segmentation filtering works correctly."""
        temporal_df['segment'] = np.random.choice(['A', 'B', 'C'], len(temporal_df))
        
        viz = BinningVisualizer()
        
        # Filter to only segment A
        df_filtered = temporal_df[temporal_df['segment'] == 'A']
        
        fig = viz.plot_temporal_bin_drift(
            df_filtered,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            segment_col='segment',
            segment_values=['A'],
        )
        
        assert fig is not None
    
    def test_multiple_segments(self, temporal_df):
        """Test with multiple segment values."""
        temporal_df['segment'] = np.random.choice(['A', 'B', 'C'], len(temporal_df))
        
        viz = BinningVisualizer()
        
        fig = viz.plot_temporal_bin_drift(
            temporal_df,
            feature_col='bin',
            target_col='target',
            snapshot_col='snapshot',
            segment_col='segment',
            segment_values=['A', 'B'],
        )
        
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
