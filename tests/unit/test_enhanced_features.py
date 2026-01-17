"""
Tests for enhanced feature engineering functionality.

Tests for:
- Temporal trend features
- Categorical encoding
- Feature validation
- Dependency graphs
- Feature registry
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from cr_score.features.enhanced_features import (
    DependencyGraph,
    FeatureValidator,
    CategoricalEncoder,
    TemporalTrendFeatures,
)


class TestDependencyGraph:
    """Test dependency graph functionality."""
    
    def test_add_feature(self):
        """Test adding features to graph."""
        graph = DependencyGraph()
        graph.add_feature("feature_a", [])
        graph.add_feature("feature_b", ["feature_a"])
        
        assert "feature_a" in graph.graph
        assert "feature_b" in graph.graph
        assert "feature_a" in graph.graph["feature_b"]
    
    def test_topological_sort_simple(self):
        """Test simple topological sort."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["b"])
        
        result = graph.topological_sort()
        
        # a must come before b, b must come before c
        assert result.index("a") < result.index("b")
        assert result.index("b") < result.index("c")
    
    def test_topological_sort_complex(self):
        """Test complex dependency ordering."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", [])
        graph.add_feature("c", ["a", "b"])
        graph.add_feature("d", ["c"])
        
        result = graph.topological_sort()
        
        # a and b must come before c
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("c")
        # c must come before d
        assert result.index("c") < result.index("d")
    
    def test_detect_cycle(self):
        """Test cycle detection."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["c"])
        graph.add_feature("c", ["a"])  # Creates cycle
        
        cycle = graph.detect_cycle()
        assert cycle is not None
        assert len(cycle) > 0
    
    def test_topological_sort_with_cycle_fails(self):
        """Test that topological sort fails with cycle."""
        graph = DependencyGraph()
        graph.add_feature("a", ["b"])
        graph.add_feature("b", ["a"])
        
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()
    
    def test_get_dependencies(self):
        """Test getting all dependencies."""
        graph = DependencyGraph()
        graph.add_feature("a", [])
        graph.add_feature("b", ["a"])
        graph.add_feature("c", ["b"])
        
        deps = graph.get_dependencies("c")
        assert "a" in deps
        assert "b" in deps


class TestFeatureValidator:
    """Test feature validation functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for validation."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000) * 10 + 50,
            'feature_3': [1] * 1000,  # Zero variance
            'feature_4': np.concatenate([np.random.randn(900), [np.nan] * 100]),  # 10% missing
        })
    
    def test_validate_basic_metrics(self, sample_df):
        """Test basic validation metrics."""
        validator = FeatureValidator()
        results = validator.validate_features(sample_df)
        
        assert 'feature_1' in results
        assert 'missing_rate' in results['feature_1']
        assert 'unique_count' in results['feature_1']
        assert 'zero_variance' in results['feature_1']
    
    def test_validate_statistical_metrics(self, sample_df):
        """Test statistical metrics."""
        validator = FeatureValidator()
        results = validator.validate_features(sample_df)
        
        feature_1_results = results['feature_1']
        assert 'mean' in feature_1_results
        assert 'std' in feature_1_results
        assert 'min' in feature_1_results
        assert 'max' in feature_1_results
        assert 'p01' in feature_1_results
        assert 'p99' in feature_1_results
    
    def test_zero_variance_detection(self, sample_df):
        """Test zero variance detection."""
        validator = FeatureValidator()
        results = validator.validate_features(sample_df)
        
        assert results['feature_3']['zero_variance'] is True
        assert results['feature_1']['zero_variance'] is False
    
    def test_missing_rate_calculation(self, sample_df):
        """Test missing rate calculation."""
        validator = FeatureValidator()
        results = validator.validate_features(sample_df)
        
        assert results['feature_4']['missing_rate'] == pytest.approx(0.1, abs=0.01)
        assert results['feature_1']['missing_rate'] == 0
    
    def test_hard_fail_thresholds(self, sample_df):
        """Test hard fail thresholds."""
        validator = FeatureValidator(hard_fail_thresholds={'missing_rate': 0.05})
        results = validator.validate_features(sample_df)
        
        assert results['feature_4']['status'] == 'FAIL'
        assert len(results['feature_4']['failures']) > 0
    
    def test_warning_thresholds(self, sample_df):
        """Test warning thresholds."""
        validator = FeatureValidator(warning_thresholds={'missing_rate': 0.05})
        results = validator.validate_features(sample_df)
        
        assert results['feature_4']['status'] == 'WARN'
        assert len(results['feature_4']['warnings']) > 0
    
    def test_to_dataframe(self, sample_df):
        """Test converting results to DataFrame."""
        validator = FeatureValidator()
        validator.validate_features(sample_df)
        
        df = validator.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert 'feature' in df.columns
        assert len(df) == 4  # 4 features
    
    def test_export_csv(self, sample_df):
        """Test exporting to CSV."""
        validator = FeatureValidator()
        validator.validate_features(sample_df)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            path = f.name
        
        try:
            validator.export_csv(path)
            assert os.path.exists(path)
            
            # Read back
            df = pd.read_csv(path)
            assert len(df) == 4
        finally:
            if os.path.exists(path):
                os.unlink(path)
    
    def test_compute_psi(self):
        """Test PSI computation."""
        validator = FeatureValidator()
        
        np.random.seed(42)
        baseline = pd.Series(np.random.randn(1000))
        current = pd.Series(np.random.randn(1000))  # Similar distribution
        
        psi = validator.compute_psi(baseline, current)
        assert psi < 0.1  # Should be low for similar distributions
        
        # Different distribution
        current_diff = pd.Series(np.random.randn(1000) * 2 + 5)
        psi_diff = validator.compute_psi(baseline, current_diff)
        assert psi_diff > psi  # Should be higher


class TestCategoricalEncoder:
    """Test categorical encoding functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with categorical data."""
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'A', 'D', 'D'],
            'target': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            'value': [10, 20, 30, 15, 25, 35, 12, 18, 22, 28],
        })
    
    def test_freq_encoding(self, sample_df):
        """Test frequency encoding."""
        encoder = CategoricalEncoder()
        result = encoder.freq_encoding(sample_df, 'category')
        
        assert 'category_freq' in result.columns
        # 'A' appears 4 times out of 10
        assert result[result['category'] == 'A']['category_freq'].iloc[0] == pytest.approx(0.4)
    
    def test_freq_encoding_custom_output(self, sample_df):
        """Test frequency encoding with custom output name."""
        encoder = CategoricalEncoder()
        result = encoder.freq_encoding(sample_df, 'category', output_col='cat_frequency')
        
        assert 'cat_frequency' in result.columns
    
    def test_target_mean_encoding(self, sample_df):
        """Test target mean encoding."""
        encoder = CategoricalEncoder()
        result = encoder.target_mean_encoding(sample_df, 'category', 'target')
        
        assert 'category_target_mean' in result.columns
        
        # 'A' has target values [1, 1, 1, 1], so mean should be 1.0
        a_mean = result[result['category'] == 'A']['category_target_mean'].iloc[0]
        assert a_mean > 0.9  # With smoothing, should be close to 1.0
    
    def test_target_mean_encoding_with_smoothing(self, sample_df):
        """Test target mean encoding with smoothing parameter."""
        encoder = CategoricalEncoder()
        result = encoder.target_mean_encoding(
            sample_df, 'category', 'target',
            smoothing=10.0  # Heavy smoothing
        )
        
        # With heavy smoothing, values should be closer to global mean
        global_mean = sample_df['target'].mean()
        a_mean = result[result['category'] == 'A']['category_target_mean'].iloc[0]
        
        # Should be between category mean and global mean
        assert 0.4 < a_mean < 1.0
    
    def test_rare_grouping(self, sample_df):
        """Test rare category grouping."""
        encoder = CategoricalEncoder()
        result = encoder.rare_grouping(sample_df, 'category', threshold=0.25)
        
        assert 'category_grouped' in result.columns
        
        # 'D' appears 2 times (20%), threshold is 25%
        # 'D' should be grouped as RARE
        d_grouped = result[result['category'] == 'D']['category_grouped'].iloc[0]
        assert d_grouped == 'RARE'
        
        # 'A' appears 4 times (40%), should NOT be grouped
        a_grouped = result[result['category'] == 'A']['category_grouped'].iloc[0]
        assert a_grouped == 'A'
    
    def test_export_mappings(self, sample_df):
        """Test exporting encoding mappings."""
        encoder = CategoricalEncoder()
        encoder.freq_encoding(sample_df, 'category')
        encoder.target_mean_encoding(sample_df, 'category', 'target')
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            path = f.name
        
        try:
            encoder.export_mappings(path)
            assert os.path.exists(path)
            
            # Read back
            import json
            with open(path) as f:
                mappings = json.load(f)
            
            assert 'category_freq' in mappings
            assert 'category_target_mean' in mappings
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestTemporalTrendFeatures:
    """Test temporal trend features."""
    
    @pytest.fixture
    def time_series_df(self):
        """Create time series DataFrame."""
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        return pd.DataFrame({
            'customer_id': ['CUST_A'] * 12,
            'date': dates,
            'balance': [1000, 1100, 1200, 1150, 1300, 1400, 1350, 1450, 1500, 1550, 1600, 1650],
            'payment': [100, 110, 120, 115, 130, 140, 135, 145, 150, 155, 160, 165],
        })
    
    def test_delta(self, time_series_df):
        """Test delta calculation."""
        trend = TemporalTrendFeatures()
        result = trend.delta(time_series_df, 'balance', time_col='date')
        
        assert 'balance_delta' in result.columns
        # Second value should be 1100 - 1000 = 100
        assert result['balance_delta'].iloc[1] == 100
    
    def test_delta_with_periods(self, time_series_df):
        """Test delta with multiple periods."""
        trend = TemporalTrendFeatures()
        result = trend.delta(time_series_df, 'balance', time_col='date', periods=2)
        
        # Third value should be 1200 - 1000 = 200
        assert result['balance_delta'].iloc[2] == 200
    
    def test_pct_change(self, time_series_df):
        """Test percent change calculation."""
        trend = TemporalTrendFeatures()
        result = trend.pct_change(time_series_df, 'balance', time_col='date')
        
        assert 'balance_pct_change' in result.columns
        # Second value should be (1100 - 1000) / 1000 = 0.1
        assert result['balance_pct_change'].iloc[1] == pytest.approx(0.1)
    
    def test_momentum(self, time_series_df):
        """Test momentum calculation."""
        trend = TemporalTrendFeatures()
        result = trend.momentum(time_series_df, 'balance', time_col='date', window=3)
        
        assert 'balance_momentum_3' in result.columns
        # Momentum should be current value minus rolling mean
        assert not result['balance_momentum_3'].isna().all()
    
    def test_volatility_std(self, time_series_df):
        """Test volatility (std) calculation."""
        trend = TemporalTrendFeatures()
        result = trend.volatility(time_series_df, 'balance', time_col='date', window=3, method='std')
        
        assert 'balance_volatility_3' in result.columns
        # Volatility should be positive
        assert (result['balance_volatility_3'].dropna() >= 0).all()
    
    def test_volatility_cv(self, time_series_df):
        """Test volatility (cv) calculation."""
        trend = TemporalTrendFeatures()
        result = trend.volatility(time_series_df, 'balance', time_col='date', window=3, method='cv')
        
        assert 'balance_volatility_3' in result.columns
        # CV should be ratio of std to mean
        assert not result['balance_volatility_3'].isna().all()
    
    def test_trend_slope(self, time_series_df):
        """Test trend slope calculation."""
        trend = TemporalTrendFeatures()
        result = trend.trend_slope(time_series_df, 'balance', time_col='date', window=6)
        
        assert 'balance_trend_slope_6' in result.columns
        # Balance is generally increasing, so slope should be mostly positive
        slopes = result['balance_trend_slope_6'].dropna()
        assert slopes.mean() > 0
    
    def test_rolling_rank(self, time_series_df):
        """Test rolling rank calculation."""
        trend = TemporalTrendFeatures()
        result = trend.rolling_rank(time_series_df, 'balance', time_col='date', window=5, pct=True)
        
        assert 'balance_rolling_rank_5' in result.columns
        # Percentile rank should be between 0 and 1
        ranks = result['balance_rolling_rank_5'].dropna()
        assert (ranks >= 0).all() and (ranks <= 1).all()
    
    def test_minmax_range(self, time_series_df):
        """Test minmax range calculation."""
        trend = TemporalTrendFeatures()
        result = trend.minmax_range(time_series_df, 'balance', time_col='date', window=5)
        
        assert 'balance_minmax_range_5' in result.columns
        # Range should be positive
        ranges = result['balance_minmax_range_5'].dropna()
        assert (ranges >= 0).all()
    
    def test_with_grouping(self):
        """Test temporal features with grouping."""
        df = pd.DataFrame({
            'customer_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'date': pd.date_range('2023-01-01', periods=6, freq='M'),
            'balance': [1000, 1100, 1200, 2000, 2100, 2200],
        })
        
        trend = TemporalTrendFeatures()
        result = trend.delta(df, 'balance', time_col='date', group_cols=['customer_id'])
        
        # First delta for each customer should be NaN
        assert result[result['customer_id'] == 'A']['balance_delta'].iloc[0] is pd.NA or np.isnan(result[result['customer_id'] == 'A']['balance_delta'].iloc[0])
        assert result[result['customer_id'] == 'B']['balance_delta'].iloc[0] is pd.NA or np.isnan(result[result['customer_id'] == 'B']['balance_delta'].iloc[0])
        
        # Second values should be 100 for both
        assert result[result['customer_id'] == 'A']['balance_delta'].iloc[1] == 100
        assert result[result['customer_id'] == 'B']['balance_delta'].iloc[1] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
