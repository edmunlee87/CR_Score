"""Spark layer for large-scale data processing."""

from cr_score.spark.session import SparkSessionFactory
from cr_score.spark.optimization import SparkCacheManager, PartitionOptimizer
from cr_score.spark.metrics import SparkExecutionMetrics, PerformanceProfiler

__all__ = [
    "SparkSessionFactory",
    "SparkCacheManager",
    "PartitionOptimizer",
    "SparkExecutionMetrics",
    "PerformanceProfiler",
]
