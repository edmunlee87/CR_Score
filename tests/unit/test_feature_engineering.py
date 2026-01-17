"""
Tests for feature engineering toolkit.
"""

import numpy as np
import pandas as pd
import pytest

from cr_score.features.engineering import (
    AggregationType,
    TimeWindow,
    FeatureRecipe,
    FeatureEngineeringConfig,
    PandasFeatureEngineer,
    create_feature_engineer,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'date': pd.date_range('2024-01-01', periods=9, freq='M'),
        'balance': [1000, 1500, 2000, 500, 600, 700, 3000, 2800, 2500],
        'credit_limit': [5000, 5000, 5000, 2000, 2000, 2000, 10000, 10000, 10000],
        'days_past_due': [0, 30, 60, 0, 0, 15, 0, 0, 0],
        'payment_amount': [100, 150, 200, 50, 60, 70, 300, 280, 250],
    })
    
    return df


@pytest.fixture
def time_series_df():
    """Create time series DataFrame for window testing."""
    df = pd.DataFrame({
        'customer_id': [1] * 12,
        'date': pd.date_range('2023-01-01', periods=12, freq='M'),
        'balance': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],
        'dpd': [0, 0, 30, 0, 0, 0, 60, 0, 0, 0, 0, 0],
    })
    return df


class TestFeatureRecipe:
    """Test FeatureRecipe class."""
    
    def test_recipe_creation(self):
        """Test basic recipe creation."""
        recipe = FeatureRecipe(
            name="max_dpd_3m",
            source_cols="dpd",
            operation=AggregationType.MAX,
            window=TimeWindow.LAST_3M,
            description="Max DPD in last 3 months"
        )
        
        assert recipe.name == "max_dpd_3m"
        assert recipe.source_cols == ["dpd"]
        assert recipe.operation == AggregationType.MAX
        assert recipe.window == TimeWindow.LAST_3M
    
    def test_recipe_with_multiple_sources(self):
        """Test recipe with multiple source columns."""
        recipe = FeatureRecipe(
            name="util_ratio",
            source_cols=["balance", "credit_limit"],
            operation="ratio"
        )
        
        assert len(recipe.source_cols) == 2
        assert "balance" in recipe.source_cols


class TestFeatureEngineeringConfig:
    """Test FeatureEngineeringConfig class."""
    
    def test_config_creation(self):
        """Test config creation."""
        recipes = [
            FeatureRecipe("max_dpd", "dpd", AggregationType.MAX),
            FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            id_col="customer_id",
            time_col="date",
            group_cols=["customer_id"]
        )
        
        assert len(config.recipes) == 2
        assert config.id_col == "customer_id"
        assert config.time_col == "date"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "recipes": [
                {
                    "name": "max_dpd",
                    "source_cols": "dpd",
                    "operation": "max",
                },
                {
                    "name": "util_ratio",
                    "source_cols": ["balance", "limit"],
                    "operation": "ratio",
                }
            ],
            "id_col": "customer_id",
            "group_cols": ["customer_id"],
        }
        
        config = FeatureEngineeringConfig.from_dict(config_dict)
        
        assert len(config.recipes) == 2
        assert config.recipes[0].name == "max_dpd"
        assert config.id_col == "customer_id"


class TestPandasFeatureEngineer:
    """Test PandasFeatureEngineer class."""
    
    def test_single_aggregation(self, sample_df):
        """Test creating single aggregation feature."""
        engineer = PandasFeatureEngineer()
        
        result = engineer.create_aggregation(
            sample_df,
            "max_balance",
            "balance",
            AggregationType.MAX,
            group_by="customer_id"
        )
        
        assert "max_balance" in result.columns
        assert result[result['customer_id'] == 1]['max_balance'].iloc[0] == 2000
        assert result[result['customer_id'] == 2]['max_balance'].iloc[0] == 700
    
    def test_ratio_feature(self, sample_df):
        """Test creating ratio feature."""
        engineer = PandasFeatureEngineer()
        
        result = engineer.create_ratio(
            sample_df,
            "utilization",
            "balance",
            "credit_limit"
        )
        
        assert "utilization" in result.columns
        assert result['utilization'].iloc[0] == pytest.approx(1000 / 5000)
    
    def test_batch_mode(self, sample_df):
        """Test batch mode with config."""
        recipes = [
            FeatureRecipe("max_dpd", "days_past_due", AggregationType.MAX),
            FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
            FeatureRecipe("util_ratio", ["balance", "credit_limit"], "ratio"),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"]
        )
        
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "max_dpd" in result.columns
        assert "avg_balance" in result.columns
        assert "util_ratio" in result.columns
        assert len(engineer.created_features_) == 3
    
    def test_time_window_filtering(self, time_series_df):
        """Test time window filtering."""
        recipes = [
            FeatureRecipe(
                "max_dpd_3m",
                "dpd",
                AggregationType.MAX,
                window=TimeWindow.LAST_3M
            ),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"],
            time_col="date"
        )
        
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(time_series_df)
        
        assert "max_dpd_3m" in result.columns
        # Last 3 months from most recent should capture last values
        assert result['max_dpd_3m'].iloc[-1] == 0  # No DPD in last 3 months
    
    def test_difference_feature(self, sample_df):
        """Test difference feature."""
        recipes = [
            FeatureRecipe(
                "available_credit",
                ["credit_limit", "balance"],
                "difference"
            ),
        ]
        
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "available_credit" in result.columns
        assert result['available_credit'].iloc[0] == 5000 - 1000
    
    def test_log_transformation(self, sample_df):
        """Test log transformation."""
        recipes = [
            FeatureRecipe(
                "log_balance",
                "balance",
                "log",
                params={"add_one": True}
            ),
        ]
        
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "log_balance" in result.columns
        assert result['log_balance'].iloc[0] == pytest.approx(np.log1p(1000))
    
    def test_sqrt_transformation(self, sample_df):
        """Test square root transformation."""
        recipes = [
            FeatureRecipe("sqrt_balance", "balance", "sqrt"),
        ]
        
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "sqrt_balance" in result.columns
        assert result['sqrt_balance'].iloc[0] == pytest.approx(np.sqrt(1000))
    
    def test_clip_transformation(self, sample_df):
        """Test clipping transformation."""
        recipes = [
            FeatureRecipe(
                "clipped_dpd",
                "days_past_due",
                "clip",
                params={"lower": 0, "upper": 30}
            ),
        ]
        
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "clipped_dpd" in result.columns
        assert result['clipped_dpd'].max() == 30  # Clipped at 30
    
    def test_multiple_aggregation_types(self, sample_df):
        """Test multiple aggregation types."""
        recipes = [
            FeatureRecipe("max_balance", "balance", AggregationType.MAX),
            FeatureRecipe("min_balance", "balance", AggregationType.MIN),
            FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
            FeatureRecipe("std_balance", "balance", AggregationType.STD),
            FeatureRecipe("sum_balance", "balance", AggregationType.SUM),
            FeatureRecipe("count_txn", "balance", AggregationType.COUNT),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"]
        )
        
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert all(f"max_balance" in result.columns for f in 
                  ["max_balance", "min_balance", "avg_balance", "std_balance", "sum_balance", "count_txn"])
    
    def test_rolling_feature(self, time_series_df):
        """Test rolling window feature."""
        engineer = PandasFeatureEngineer()
        
        result = engineer.create_rolling_feature(
            time_series_df,
            "balance_ma3",
            "balance",
            window=3,
            operation=AggregationType.MEAN,
            group_by="customer_id"
        )
        
        assert "balance_ma3" in result.columns
        # Third row should be average of first 3 values
        expected = (1000 + 1100 + 1200) / 3
        assert result['balance_ma3'].iloc[2] == pytest.approx(expected)
    
    def test_range_aggregation(self, sample_df):
        """Test range aggregation (max - min)."""
        recipes = [
            FeatureRecipe("balance_range", "balance", AggregationType.RANGE),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"]
        )
        
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        assert "balance_range" in result.columns
        # Customer 1: 2000 - 1000 = 1000
        customer_1_range = result[result['customer_id'] == 1]['balance_range'].iloc[0]
        assert customer_1_range == 1000
    
    def test_division_by_zero(self, sample_df):
        """Test that division by zero is handled properly."""
        # Add a row with zero denominator
        df = sample_df.copy()
        df.loc[0, 'credit_limit'] = 0
        
        engineer = PandasFeatureEngineer()
        result = engineer.create_ratio(df, "util", "balance", "credit_limit")
        
        assert pd.isna(result['util'].iloc[0])  # Should be NaN, not error
    
    def test_empty_config_error(self):
        """Test that transform without config raises error."""
        engineer = PandasFeatureEngineer()
        
        with pytest.raises(ValueError, match="Config required"):
            engineer.transform(pd.DataFrame({'a': [1, 2, 3]}))
    
    def test_invalid_operation(self, sample_df):
        """Test that invalid operation raises error."""
        recipes = [
            FeatureRecipe("invalid", "balance", "invalid_operation"),
        ]
        
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer = PandasFeatureEngineer(config)
        
        with pytest.raises(ValueError, match="Unknown operation"):
            engineer.fit_transform(sample_df)


class TestFactoryFunction:
    """Test create_feature_engineer factory function."""
    
    def test_create_pandas_engineer(self):
        """Test creating pandas engineer via factory."""
        engineer = create_feature_engineer(engine="pandas")
        assert isinstance(engineer, PandasFeatureEngineer)
    
    def test_create_spark_engineer(self):
        """Test creating spark engineer via factory."""
        try:
            engineer = create_feature_engineer(engine="spark")
            from cr_score.features.engineering import SparkFeatureEngineer
            assert isinstance(engineer, SparkFeatureEngineer)
        except ImportError:
            pytest.skip("PySpark not available")
    
    def test_invalid_engine(self):
        """Test that invalid engine raises error."""
        with pytest.raises(ValueError, match="Unknown engine"):
            create_feature_engineer(engine="invalid")


class TestAggregationTypes:
    """Test all aggregation types."""
    
    def test_all_aggregation_types(self, sample_df):
        """Test that all aggregation types work."""
        aggregations = [
            AggregationType.MAX,
            AggregationType.MIN,
            AggregationType.MEAN,
            AggregationType.STD,
            AggregationType.SUM,
            AggregationType.COUNT,
            AggregationType.RANGE,
            AggregationType.WORST,
        ]
        
        engineer = PandasFeatureEngineer()
        
        for agg in aggregations:
            result = engineer.create_aggregation(
                sample_df,
                f"test_{agg.value}",
                "balance",
                agg,
                group_by="customer_id"
            )
            assert f"test_{agg.value}" in result.columns


class TestTimeWindows:
    """Test time window functionality."""
    
    def test_time_window_enum_values(self):
        """Test time window enum values."""
        assert TimeWindow.LAST_1M == "last_1_month"
        assert TimeWindow.LAST_3M == "last_3_months"
        assert TimeWindow.LAST_6M == "last_6_months"
        assert TimeWindow.LAST_12M == "last_12_months"
        assert TimeWindow.LAST_24M == "last_24_months"
        assert TimeWindow.ALL_TIME == "all_time"
    
    def test_custom_window_months(self, time_series_df):
        """Test custom window in months."""
        engineer = PandasFeatureEngineer()
        
        result = engineer.create_aggregation(
            time_series_df,
            "max_balance_6m",
            "balance",
            AggregationType.MAX,
            group_by="customer_id",
            window_months=6,
            time_col="date"
        )
        
        assert "max_balance_6m" in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['a', 'b', 'c'])
        engineer = PandasFeatureEngineer()
        
        result = engineer.create_ratio(df, "ratio", "a", "b")
        assert len(result) == 0
        assert "ratio" in result.columns
    
    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'a': [10],
            'b': [5],
        })
        
        engineer = PandasFeatureEngineer()
        result = engineer.create_ratio(df, "ratio", "a", "b")
        
        assert len(result) == 1
        assert result['ratio'].iloc[0] == 2.0
    
    def test_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'customer_id': [1, 1, 1],
            'balance': [1000, np.nan, 2000],
        })
        
        engineer = PandasFeatureEngineer()
        result = engineer.create_aggregation(
            df,
            "max_balance",
            "balance",
            AggregationType.MAX,
            group_by="customer_id"
        )
        
        assert "max_balance" in result.columns
        # Should handle NaN gracefully
        assert result['max_balance'].iloc[0] == 2000


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_credit_risk_features(self, sample_df):
        """Test creating typical credit risk features."""
        recipes = [
            # Delinquency features
            FeatureRecipe("max_dpd", "days_past_due", AggregationType.MAX),
            FeatureRecipe("avg_dpd", "days_past_due", AggregationType.MEAN),
            
            # Balance features
            FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
            FeatureRecipe("balance_range", "balance", AggregationType.RANGE),
            
            # Utilization
            FeatureRecipe("utilization", ["balance", "credit_limit"], "ratio"),
            
            # Payment behavior
            FeatureRecipe("total_payments", "payment_amount", AggregationType.SUM),
            FeatureRecipe("avg_payment", "payment_amount", AggregationType.MEAN),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"]
        )
        
        engineer = PandasFeatureEngineer(config)
        result = engineer.fit_transform(sample_df)
        
        # Verify all features created
        expected_features = [
            "max_dpd", "avg_dpd", "avg_balance", "balance_range",
            "utilization", "total_payments", "avg_payment"
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Verify some values
        customer_1 = result[result['customer_id'] == 1].iloc[0]
        assert customer_1['max_dpd'] == 60
        assert customer_1['utilization'] == pytest.approx(1000 / 5000)
    
    def test_chained_transformations(self, sample_df):
        """Test chaining multiple transformations."""
        engineer = PandasFeatureEngineer()
        
        # Step 1: Create ratio
        df = engineer.create_ratio(
            sample_df,
            "utilization",
            "balance",
            "credit_limit"
        )
        
        # Step 2: Create log of utilization
        recipes = [
            FeatureRecipe("log_util", "utilization", "log", params={"add_one": True})
        ]
        config = FeatureEngineeringConfig(recipes=recipes)
        engineer2 = PandasFeatureEngineer(config)
        
        result = engineer2.fit_transform(df)
        
        assert "utilization" in result.columns
        assert "log_util" in result.columns
