"""
Spark caching utilities for intelligent memory management.

Provides smart caching strategies based on DataFrame size, usage patterns, and memory availability.
"""

from typing import Dict, List, Optional, Set
from enum import Enum

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark import StorageLevel
from pyspark.sql import SparkSession

from cr_score.core.logging import get_audit_logger


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY_ONLY = StorageLevel.MEMORY_ONLY
    MEMORY_AND_DISK = StorageLevel.MEMORY_AND_DISK
    MEMORY_ONLY_2 = StorageLevel.MEMORY_ONLY_2
    MEMORY_AND_DISK_2 = StorageLevel.MEMORY_AND_DISK_2
    DISK_ONLY = StorageLevel.DISK_ONLY


class SparkCacheManager:
    """
    Intelligent caching manager for Spark DataFrames.
    
    Automatically selects appropriate cache levels, tracks usage,
    and manages memory efficiently.
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        default_level: CacheLevel = CacheLevel.MEMORY_AND_DISK,
        max_memory_fraction: float = 0.6,
    ):
        """
        Initialize cache manager.
        
        Args:
            spark: Spark session (auto-detected if None)
            default_level: Default storage level for caching
            max_memory_fraction: Maximum fraction of executor memory to use for caching
        """
        self.spark = spark or SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("No active Spark session found. Provide spark parameter or create a session.")
        
        self.default_level = default_level
        self.max_memory_fraction = max_memory_fraction
        self.logger = get_audit_logger()
        
        # Track cached DataFrames
        self._cached_dfs: Dict[str, SparkDataFrame] = {}
        self._cache_stats: Dict[str, Dict] = {}
        self._cache_hits: Dict[str, int] = {}
        self._cache_misses: Dict[str, int] = {}
    
    def cache_if_reused(
        self,
        df: SparkDataFrame,
        name: str,
        min_reuses: int = 2,
        force: bool = False,
    ) -> SparkDataFrame:
        """
        Cache DataFrame if it will be reused multiple times.
        
        Args:
            df: DataFrame to potentially cache
            name: Name identifier for tracking
            min_reuses: Minimum number of reuses to justify caching
            force: Force caching regardless of reuse count
            
        Returns:
            Cached DataFrame (or original if not cached)
        """
        if force or self._cache_hits.get(name, 0) >= min_reuses:
            return self.persist_with_level(df, name, self.default_level)
        return df
    
    def persist_with_level(
        self,
        df: SparkDataFrame,
        name: str,
        level: Optional[CacheLevel] = None,
    ) -> SparkDataFrame:
        """
        Persist DataFrame with appropriate storage level.
        
        Args:
            df: DataFrame to persist
            name: Name identifier for tracking
            level: Storage level (uses default if None)
            
        Returns:
            Persisted DataFrame
        """
        if level is None:
            level = self._select_optimal_level(df)
        
        df_cached = df.persist(level.value)
        self._cached_dfs[name] = df_cached
        self._cache_stats[name] = {
            'level': level.name,
            'partitions': df.rdd.getNumPartitions(),
            'cached_at': self._get_timestamp(),
        }
        self._cache_hits[name] = 0
        self._cache_misses[name] = 0
        
        self.logger.info(f"Cached DataFrame '{name}' with level {level.name}")
        return df_cached
    
    def _select_optimal_level(self, df: SparkDataFrame) -> CacheLevel:
        """
        Select optimal cache level based on DataFrame characteristics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Optimal cache level
        """
        # Estimate size (rough heuristic)
        num_partitions = df.rdd.getNumPartitions()
        num_rows = df.count()
        
        # Very large DataFrames: use disk
        if num_rows > 10_000_000 or num_partitions > 1000:
            return CacheLevel.MEMORY_AND_DISK
        
        # Medium DataFrames: use memory with disk spill
        if num_rows > 1_000_000:
            return CacheLevel.MEMORY_AND_DISK
        
        # Small DataFrames: memory only
        return CacheLevel.MEMORY_ONLY
    
    def unpersist(self, name: Optional[str] = None) -> None:
        """
        Unpersist cached DataFrame(s).
        
        Args:
            name: Name of DataFrame to unpersist (None = all)
        """
        if name is None:
            # Unpersist all
            for df_name, df in self._cached_dfs.items():
                df.unpersist()
                self.logger.info(f"Unpersisted DataFrame '{df_name}'")
            self._cached_dfs.clear()
            self._cache_stats.clear()
        else:
            if name in self._cached_dfs:
                self._cached_dfs[name].unpersist()
                del self._cached_dfs[name]
                if name in self._cache_stats:
                    del self._cache_stats[name]
                self.logger.info(f"Unpersisted DataFrame '{name}'")
    
    def get_cache_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all cached DataFrames.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {}
        for name, df in self._cached_dfs.items():
            try:
                # Get storage info from Spark UI
                storage_info = self._get_storage_info(df)
                stats[name] = {
                    **self._cache_stats.get(name, {}),
                    'hits': self._cache_hits.get(name, 0),
                    'misses': self._cache_misses.get(name, 0),
                    'storage_info': storage_info,
                }
            except Exception as e:
                self.logger.warning(f"Could not get stats for '{name}': {e}")
                stats[name] = self._cache_stats.get(name, {})
        
        return stats
    
    def _get_storage_info(self, df: SparkDataFrame) -> Dict:
        """Get storage information from Spark."""
        try:
            # Access Spark context storage info
            rdd_id = df.rdd.id
            status = self.spark.sparkContext.statusTracker().getRDDInfo(rdd_id)
            if status:
                return {
                    'memory_size': status.memorySize,
                    'disk_size': status.diskSize,
                    'num_cached_partitions': status.numCachedPartitions,
                }
        except Exception:
            pass
        return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def record_cache_hit(self, name: str) -> None:
        """Record a cache hit."""
        self._cache_hits[name] = self._cache_hits.get(name, 0) + 1
    
    def record_cache_miss(self, name: str) -> None:
        """Record a cache miss."""
        self._cache_misses[name] = self._cache_misses.get(name, 0) + 1
    
    def clear_all(self) -> None:
        """Clear all cached DataFrames and statistics."""
        self.unpersist()
        self._cache_hits.clear()
        self._cache_misses.clear()
        self.logger.info("Cleared all cache and statistics")
