"""
Tests for Spark optimization utilities.
"""

import pytest
import pandas as pd
import numpy as np

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import functions as F
    from pyspark import StorageLevel
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None
    F = None
    StorageLevel = None

if PYSPARK_AVAILABLE:
    from cr_score.spark.optimization import SparkCacheManager, PartitionOptimizer, CacheLevel


@pytest.fixture
def spark_session():
    """Create Spark session for testing."""
    if not PYSPARK_AVAILABLE:
        pytest.skip("PySpark not available")
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_spark_data(spark_session):
    """Create sample Spark DataFrame."""
    data = pd.DataFrame({
        'customer_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'date': pd.date_range('2024-01-01', periods=9, freq='M'),
        'balance': [1000, 1200, 1100, 2000, 2100, 2050, 3000, 3100, 3050],
        'utilization': [0.3, 0.35, 0.32, 0.5, 0.55, 0.52, 0.6, 0.65, 0.62],
    })
    return spark_session.createDataFrame(data)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestSparkCacheManager:
    """Test SparkCacheManager."""
    
    def test_init(self, spark_session):
        """Test initialization."""
        cache_mgr = SparkCacheManager(spark_session)
        assert cache_mgr.spark == spark_session
        assert cache_mgr.default_level == CacheLevel.MEMORY_AND_DISK
    
    def test_persist_with_level(self, spark_session, sample_spark_data):
        """Test persisting with storage level."""
        cache_mgr = SparkCacheManager(spark_session)
        df_cached = cache_mgr.persist_with_level(sample_spark_data, "test_df", CacheLevel.MEMORY_ONLY)
        
        assert "test_df" in cache_mgr._cached_dfs
        assert cache_mgr._cache_stats["test_df"]["level"] == "MEMORY_ONLY"
    
    def test_cache_if_reused_force(self, spark_session, sample_spark_data):
        """Test force caching."""
        cache_mgr = SparkCacheManager(spark_session)
        df_cached = cache_mgr.cache_if_reused(sample_spark_data, "test_df", force=True)
        
        assert "test_df" in cache_mgr._cached_dfs
    
    def test_cache_if_reused_min_reuses(self, spark_session, sample_spark_data):
        """Test caching based on reuse count."""
        cache_mgr = SparkCacheManager(spark_session)
        
        # Record hits to trigger caching
        cache_mgr._cache_hits["test_df"] = 3
        
        df_cached = cache_mgr.cache_if_reused(sample_spark_data, "test_df", min_reuses=2)
        assert "test_df" in cache_mgr._cached_dfs
    
    def test_unpersist(self, spark_session, sample_spark_data):
        """Test unpersisting."""
        cache_mgr = SparkCacheManager(spark_session)
        cache_mgr.persist_with_level(sample_spark_data, "test_df")
        
        assert "test_df" in cache_mgr._cached_dfs
        cache_mgr.unpersist("test_df")
        assert "test_df" not in cache_mgr._cached_dfs
    
    def test_unpersist_all(self, spark_session, sample_spark_data):
        """Test unpersisting all."""
        cache_mgr = SparkCacheManager(spark_session)
        cache_mgr.persist_with_level(sample_spark_data, "test_df1")
        cache_mgr.persist_with_level(sample_spark_data, "test_df2")
        
        assert len(cache_mgr._cached_dfs) == 2
        cache_mgr.unpersist()
        assert len(cache_mgr._cached_dfs) == 0
    
    def test_get_cache_stats(self, spark_session, sample_spark_data):
        """Test getting cache statistics."""
        cache_mgr = SparkCacheManager(spark_session)
        cache_mgr.persist_with_level(sample_spark_data, "test_df")
        
        stats = cache_mgr.get_cache_stats()
        assert "test_df" in stats
        assert "level" in stats["test_df"]
    
    def test_record_cache_hit_miss(self, spark_session):
        """Test recording cache hits and misses."""
        cache_mgr = SparkCacheManager(spark_session)
        cache_mgr.record_cache_hit("test_df")
        cache_mgr.record_cache_miss("test_df")
        
        assert cache_mgr._cache_hits["test_df"] == 1
        assert cache_mgr._cache_misses["test_df"] == 1
    
    def test_clear_all(self, spark_session, sample_spark_data):
        """Test clearing all cache."""
        cache_mgr = SparkCacheManager(spark_session)
        cache_mgr.persist_with_level(sample_spark_data, "test_df")
        cache_mgr.record_cache_hit("test_df")
        
        cache_mgr.clear_all()
        assert len(cache_mgr._cached_dfs) == 0
        assert len(cache_mgr._cache_hits) == 0


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestPartitionOptimizer:
    """Test PartitionOptimizer."""
    
    def test_init(self, spark_session):
        """Test initialization."""
        optimizer = PartitionOptimizer(spark_session)
        assert optimizer.spark == spark_session
        assert optimizer.target_partition_size_mb == 128.0
    
    def test_optimize_partitions(self, spark_session, sample_spark_data):
        """Test partition optimization."""
        optimizer = PartitionOptimizer(spark_session, target_partition_size_mb=1.0)
        
        original_partitions = sample_spark_data.rdd.getNumPartitions()
        df_optimized = optimizer.optimize_partitions(sample_spark_data)
        
        # Should return a DataFrame (may or may not repartition depending on size)
        assert isinstance(df_optimized, SparkDataFrame)
    
    def test_coalesce_if_needed(self, spark_session, sample_spark_data):
        """Test coalescing."""
        optimizer = PartitionOptimizer(spark_session)
        
        # Repartition to many partitions first
        df_many = sample_spark_data.repartition(100)
        assert df_many.rdd.getNumPartitions() == 100
        
        # Coalesce
        df_coalesced = optimizer.coalesce_if_needed(df_many, max_partitions=10)
        assert df_coalesced.rdd.getNumPartitions() <= 10
    
    def test_detect_skew(self, spark_session):
        """Test skew detection."""
        # Create skewed data
        data = pd.DataFrame({
            'customer_id': [1] * 1000 + [2] * 10 + [3] * 10,
            'value': np.random.randn(1020),
        })
        df = spark_session.createDataFrame(data)
        
        optimizer = PartitionOptimizer(spark_session, skew_threshold=2.0)
        skew_info = optimizer.detect_skew(df, "customer_id", sample_fraction=1.0)
        
        assert "skewed" in skew_info
        assert "ratio" in skew_info
        # Should detect skew (customer_id=1 has 1000, others have 10)
        assert skew_info["skewed"] is True
    
    def test_add_salting(self, spark_session, sample_spark_data):
        """Test adding salt."""
        optimizer = PartitionOptimizer(spark_session)
        df_salted, salted_col = optimizer.add_salting(
            sample_spark_data,
            "customer_id",
            num_salts=5
        )
        
        assert salted_col == "customer_id_salted"
        assert salted_col in df_salted.columns
        # Check that salted column contains original key + salt
        salted_values = df_salted.select(salted_col).distinct().collect()
        assert len(salted_values) > 0
    
    def test_remove_salting(self, spark_session):
        """Test removing salt."""
        optimizer = PartitionOptimizer(spark_session)
        
        # Create salted DataFrame
        df = spark_session.createDataFrame(pd.DataFrame({
            'customer_id_salted': ['1_0', '1_1', '2_0', '2_1'],
            'value': [1, 2, 3, 4],
        }))
        
        df_clean = optimizer.remove_salting(df, "customer_id_salted", "customer_id")
        
        assert "customer_id" in df_clean.columns
        assert "customer_id_salted" not in df_clean.columns
    
    def test_optimize_for_join(self, spark_session, sample_spark_data):
        """Test optimization for join."""
        optimizer = PartitionOptimizer(spark_session)
        
        df_optimized = optimizer.optimize_for_join(sample_spark_data, "customer_id")
        
        assert isinstance(df_optimized, SparkDataFrame)
        # Should be partitioned by join key
        assert df_optimized.rdd.getNumPartitions() > 0


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestCacheLevel:
    """Test CacheLevel enum."""
    
    def test_cache_level_values(self):
        """Test cache level enum values."""
        assert CacheLevel.MEMORY_ONLY.value == StorageLevel.MEMORY_ONLY
        assert CacheLevel.MEMORY_AND_DISK.value == StorageLevel.MEMORY_AND_DISK
        assert CacheLevel.DISK_ONLY.value == StorageLevel.DISK_ONLY
