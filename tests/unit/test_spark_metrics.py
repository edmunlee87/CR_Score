"""
Tests for Spark metrics collection utilities.
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame as SparkDataFrame
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None

if PYSPARK_AVAILABLE:
    from cr_score.spark.metrics import SparkExecutionMetrics, PerformanceProfiler


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
        'customer_id': [1, 1, 1, 2, 2, 2],
        'date': pd.date_range('2024-01-01', periods=6, freq='M'),
        'balance': [1000, 1200, 1100, 2000, 2100, 2050],
        'utilization': [0.3, 0.35, 0.32, 0.5, 0.55, 0.52],
    })
    return spark_session.createDataFrame(data)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestSparkExecutionMetrics:
    """Test SparkExecutionMetrics."""
    
    def test_init(self, spark_session):
        """Test initialization."""
        metrics = SparkExecutionMetrics(spark_session)
        assert metrics.spark == spark_session
        assert metrics.sc == spark_session.sparkContext
    
    def test_get_spark_config(self, spark_session):
        """Test getting Spark configuration."""
        metrics = SparkExecutionMetrics(spark_session)
        config = metrics.get_spark_config()
        
        assert isinstance(config, dict)
        assert len(config) > 0
    
    def test_get_executor_metrics(self, spark_session):
        """Test getting executor metrics."""
        metrics = SparkExecutionMetrics(spark_session)
        executor_metrics = metrics.get_executor_metrics()
        
        assert isinstance(executor_metrics, list)
        # In local mode, should have at least one executor
        assert len(executor_metrics) >= 0
    
    def test_track_execution_success(self, spark_session, sample_spark_data):
        """Test tracking successful execution."""
        metrics = SparkExecutionMetrics(spark_session)
        
        def test_func(df):
            return df.count()
        
        result, metrics_data = metrics.track_execution(
            "test_operation",
            test_func,
            sample_spark_data
        )
        
        assert result == 6  # 6 rows
        assert metrics_data["operation"] == "test_operation"
        assert metrics_data["status"] == "success"
        assert "duration_seconds" in metrics_data
    
    def test_track_execution_failure(self, spark_session):
        """Test tracking failed execution."""
        metrics = SparkExecutionMetrics(spark_session)
        
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            metrics.track_execution("failing_operation", failing_func)
        
        # Check that failure was recorded
        history = metrics.get_metrics_history()
        assert len(history) > 0
        assert history[-1]["status"] == "failed"
    
    def test_get_metrics_history(self, spark_session, sample_spark_data):
        """Test getting metrics history."""
        metrics = SparkExecutionMetrics(spark_session)
        
        def test_func(df):
            return df.count()
        
        metrics.track_execution("op1", test_func, sample_spark_data)
        metrics.track_execution("op2", test_func, sample_spark_data)
        
        history = metrics.get_metrics_history()
        assert len(history) == 2
        assert history[0]["operation"] == "op1"
        assert history[1]["operation"] == "op2"
    
    def test_export_metrics(self, spark_session, sample_spark_data):
        """Test exporting metrics."""
        metrics = SparkExecutionMetrics(spark_session)
        
        def test_func(df):
            return df.count()
        
        metrics.track_execution("test_operation", test_func, sample_spark_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            metrics.export_metrics(export_path)
            
            # Verify file was created and contains data
            assert os.path.exists(export_path)
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert "spark_config" in data
            assert "execution_history" in data
            assert len(data["execution_history"]) > 0
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")
class TestPerformanceProfiler:
    """Test PerformanceProfiler."""
    
    def test_init(self, spark_session):
        """Test initialization."""
        profiler = PerformanceProfiler(spark_session)
        assert profiler.spark == spark_session
        assert profiler.metrics is not None
    
    def test_profile_operation_success(self, spark_session, sample_spark_data):
        """Test profiling successful operation."""
        profiler = PerformanceProfiler(spark_session)
        
        def test_func(df):
            return df.count()
        
        profile = profiler.profile_operation("test_count", test_func, sample_spark_data)
        
        assert profile["operation"] == "test_count"
        assert profile["status"] == "success"
        assert profile["duration_seconds"] > 0
        assert "job_metrics" in profile
    
    def test_profile_operation_dataframe(self, spark_session, sample_spark_data):
        """Test profiling operation that returns DataFrame."""
        profiler = PerformanceProfiler(spark_session)
        
        def test_func(df):
            return df.filter("balance > 1500")
        
        profile = profiler.profile_operation("test_filter", test_func, sample_spark_data)
        
        assert profile["status"] == "success"
        assert "dataframe_metrics" in profile
        assert "num_partitions" in profile["dataframe_metrics"]
    
    def test_profile_operation_failure(self, spark_session):
        """Test profiling failed operation."""
        profiler = PerformanceProfiler(spark_session)
        
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            profiler.profile_operation("failing_op", failing_func)
        
        # Check that failure was recorded
        history = profiler.get_profile_history()
        assert len(history) > 0
        assert history[-1]["status"] == "failed"
    
    def test_compare_operations(self, spark_session, sample_spark_data):
        """Test comparing multiple operations."""
        profiler = PerformanceProfiler(spark_session)
        
        def op1(df):
            return df.count()
        
        def op2(df):
            return df.select("customer_id").distinct().count()
        
        operations = {
            "count": op1,
            "distinct_count": op2,
        }
        
        results = profiler.compare_operations(operations, sample_spark_data)
        
        assert "count" in results
        assert "distinct_count" in results
        assert "_comparison" in results
    
    def test_analyze_bottlenecks(self, spark_session, sample_spark_data):
        """Test bottleneck analysis."""
        profiler = PerformanceProfiler(spark_session)
        
        def test_func(df):
            return df.count()
        
        profile = profiler.profile_operation("test_op", test_func, sample_spark_data)
        bottlenecks = profiler.analyze_bottlenecks(profile)
        
        assert isinstance(bottlenecks, list)
        # For small operations, should have few or no bottlenecks
    
    def test_get_profile_history(self, spark_session, sample_spark_data):
        """Test getting profile history."""
        profiler = PerformanceProfiler(spark_session)
        
        def test_func(df):
            return df.count()
        
        profiler.profile_operation("op1", test_func, sample_spark_data)
        profiler.profile_operation("op2", test_func, sample_spark_data)
        
        history = profiler.get_profile_history()
        assert len(history) == 2
        assert history[0]["operation"] == "op1"
        assert history[1]["operation"] == "op2"
    
    def test_generate_report(self, spark_session, sample_spark_data):
        """Test generating performance report."""
        profiler = PerformanceProfiler(spark_session)
        
        def test_func(df):
            return df.count()
        
        profiler.profile_operation("test_op", test_func, sample_spark_data)
        
        report = profiler.generate_report()
        
        assert isinstance(report, str)
        assert "Spark Performance Profiling Report" in report
        assert "test_op" in report
    
    def test_generate_report_empty(self, spark_session):
        """Test generating report with no data."""
        profiler = PerformanceProfiler(spark_session)
        report = profiler.generate_report()
        
        assert "No profiling data available" in report
