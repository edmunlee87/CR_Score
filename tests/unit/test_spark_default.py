"""
Tests for Spark default feature engineering functionality.
"""

import pytest
import pandas as pd
import numpy as np

from cr_score.features.engineering import (
    create_feature_engineer,
    create_feature_engineer_auto,
    FeatureEngineer,
    FeatureEngineeringConfig,
    FeatureRecipe,
    AggregationType,
    PandasFeatureEngineer,
    SparkFeatureEngineer,
)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None
    F = None


@pytest.fixture
def sample_data():
    """Create sample pandas DataFrame."""
    return pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 2],
        'date': pd.date_range('2024-01-01', periods=6, freq='M'),
        'balance': [1000, 1200, 1100, 2000, 2100, 2050],
        'utilization': [0.3, 0.35, 0.32, 0.5, 0.55, 0.52],
    })


@pytest.fixture
def spark_session():
    """Create Spark session for testing."""
    if not PYSPARK_AVAILABLE:
        pytest.skip("PySpark not available")
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def spark_data(spark_session, sample_data):
    """Create Spark DataFrame from sample data."""
    return spark_session.createDataFrame(sample_data)


class TestFactoryDefault:
    """Test factory function default behavior."""
    
    def test_default_is_spark(self):
        """Test that factory defaults to Spark."""
        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = create_feature_engineer(config)
        assert isinstance(engineer, SparkFeatureEngineer)
    
    def test_explicit_pandas(self):
        """Test explicit pandas engine still works."""
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = create_feature_engineer(config, engine="pandas")
        assert isinstance(engineer, PandasFeatureEngineer)
    
    def test_explicit_spark(self, spark_session):
        """Test explicit Spark engine."""
        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = create_feature_engineer(config, engine="spark")
        assert isinstance(engineer, SparkFeatureEngineer)


class TestAutoDetection:
    """Test auto-detection functionality."""
    
    def test_auto_detect_spark_dataframe(self, spark_data):
        """Test auto-detection with Spark DataFrame."""
        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = create_feature_engineer_auto(spark_data, config)
        assert isinstance(engineer, SparkFeatureEngineer)
    
    def test_auto_detect_pandas_dataframe(self, sample_data):
        """Test auto-detection with pandas DataFrame."""
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        # With prefer_spark=False, should use pandas
        engineer = create_feature_engineer_auto(sample_data, config, prefer_spark=False)
        assert isinstance(engineer, PandasFeatureEngineer)
    
    def test_auto_detect_prefer_spark(self, sample_data):
        """Test prefer_spark parameter."""
        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        # With prefer_spark=True, should use Spark even for pandas DataFrame
        engineer = create_feature_engineer_auto(sample_data, config, prefer_spark=True)
        assert isinstance(engineer, SparkFeatureEngineer)


class TestUnifiedFeatureEngineer:
    """Test unified FeatureEngineer class."""
    
    def test_unified_auto_detect_spark(self, spark_data):
        """Test unified engineer auto-detects Spark."""
        if not PYSPARK_AVAILABLE:
            pytest.skip("PySpark not available")
        
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = FeatureEngineer(config)
        result = engineer.fit_transform(spark_data)
        
        assert engineer.detected_engine == "spark"
        assert isinstance(result, SparkDataFrame)
    
    def test_unified_auto_detect_pandas(self, sample_data):
        """Test unified engineer auto-detects pandas."""
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = FeatureEngineer(config, prefer_spark=False)
        result = engineer.fit_transform(sample_data)
        
        assert engineer.detected_engine == "pandas"
        assert isinstance(result, pd.DataFrame)
    
    def test_unified_explicit_engine(self, sample_data):
        """Test unified engineer with explicit engine."""
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = FeatureEngineer(config, engine="pandas")
        result = engineer.fit_transform(sample_data)
        
        assert engineer.detected_engine == "pandas"
        assert isinstance(result, pd.DataFrame)


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_explicit_pandas_still_works(self, sample_data):
        """Test that explicit pandas engine still works."""
        config = FeatureEngineeringConfig(
            recipes=[FeatureRecipe("test_feature", "balance", AggregationType.MAX)],
            id_col="customer_id"
        )
        
        engineer = create_feature_engineer(config, engine="pandas")
        result = engineer.fit_transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert "test_feature" in result.columns


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestSparkEnhancedFeatures:
    """Test Spark enhanced features auto-detection."""
    
    def test_temporal_features_auto_detect(self, spark_data):
        """Test temporal features auto-detect Spark."""
        from cr_score.features.enhanced_features import TemporalTrendFeatures
        
        trend = TemporalTrendFeatures()
        result = trend.delta(spark_data, "balance", time_col="date", group_cols=["customer_id"])
        
        assert isinstance(result, SparkDataFrame)
        assert "balance_delta" in result.columns
    
    def test_categorical_encoder_auto_detect(self, spark_data):
        """Test categorical encoder auto-detects Spark."""
        from cr_score.features.enhanced_features import CategoricalEncoder
        
        encoder = CategoricalEncoder()
        # Add a categorical column
        spark_data_cat = spark_data.withColumn("category", F.lit("A"))
        result = encoder.freq_encoding(spark_data_cat, "category")
        
        assert isinstance(result, SparkDataFrame)
        assert "category_freq" in result.columns
    
    def test_feature_validator_auto_detect(self, spark_data):
        """Test feature validator auto-detects Spark."""
        from cr_score.features.enhanced_features import FeatureValidator
        
        validator = FeatureValidator()
        results = validator.validate_features(spark_data, feature_list=["balance", "utilization"])
        
        assert "balance" in results
        assert "utilization" in results
        assert "missing_rate" in results["balance"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
