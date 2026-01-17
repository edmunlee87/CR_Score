"""
Spark partitioning optimization utilities.

Provides skew detection, salting strategies, and partition optimization.
"""

from typing import Dict, List, Optional, Tuple
import math

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from cr_score.core.logging import get_audit_logger


class PartitionOptimizer:
    """
    Partition optimization utilities for Spark DataFrames.
    
    Handles skew detection, salting, and partition count optimization.
    """
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        target_partition_size_mb: float = 128.0,
        skew_threshold: float = 3.0,
    ):
        """
        Initialize partition optimizer.
        
        Args:
            spark: Spark session (auto-detected if None)
            target_partition_size_mb: Target partition size in MB
            skew_threshold: Skew threshold (max/mean ratio)
        """
        self.spark = spark or SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("No active Spark session found. Provide spark parameter or create a session.")
        
        self.target_partition_size_mb = target_partition_size_mb
        self.skew_threshold = skew_threshold
        self.logger = get_audit_logger()
    
    def optimize_partitions(
        self,
        df: SparkDataFrame,
        target_size_mb: Optional[float] = None,
    ) -> SparkDataFrame:
        """
        Optimize partition count based on data size.
        
        Args:
            df: DataFrame to optimize
            target_size_mb: Target partition size (uses default if None)
            
        Returns:
            Repartitioned DataFrame
        """
        if target_size_mb is None:
            target_size_mb = self.target_partition_size_mb
        
        # Estimate data size
        current_partitions = df.rdd.getNumPartitions()
        
        # Get approximate size from Spark
        try:
            # Use Spark's size estimation
            size_bytes = self._estimate_size(df)
            size_mb = size_bytes / (1024 * 1024)
            
            # Calculate optimal partition count
            optimal_partitions = max(1, int(math.ceil(size_mb / target_size_mb)))
            
            # Clamp to reasonable range
            optimal_partitions = max(1, min(optimal_partitions, 1000))
            
            if optimal_partitions != current_partitions:
                self.logger.info(
                    f"Repartitioning from {current_partitions} to {optimal_partitions} partitions "
                    f"(estimated size: {size_mb:.1f} MB)"
                )
                return df.repartition(optimal_partitions)
            else:
                self.logger.debug(f"Partition count already optimal: {current_partitions}")
                return df
        except Exception as e:
            self.logger.warning(f"Could not estimate size, using current partitions: {e}")
            return df
    
    def coalesce_if_needed(
        self,
        df: SparkDataFrame,
        max_partitions: int = 200,
    ) -> SparkDataFrame:
        """
        Coalesce partitions if there are too many.
        
        Args:
            df: DataFrame to coalesce
            max_partitions: Maximum number of partitions
            
        Returns:
            Coalesced DataFrame
        """
        current_partitions = df.rdd.getNumPartitions()
        
        if current_partitions > max_partitions:
            self.logger.info(f"Coalescing from {current_partitions} to {max_partitions} partitions")
            return df.coalesce(max_partitions)
        
        return df
    
    def detect_skew(
        self,
        df: SparkDataFrame,
        key_col: str,
        sample_fraction: float = 0.1,
    ) -> Dict[str, float]:
        """
        Detect skew in a key column.
        
        Args:
            df: DataFrame to analyze
            key_col: Column to check for skew
            sample_fraction: Fraction of data to sample for analysis
            
        Returns:
            Dictionary with skew metrics
        """
        # Sample data for analysis
        df_sample = df.sample(False, sample_fraction, seed=42)
        
        # Count occurrences per key
        key_counts = (
            df_sample
            .groupBy(key_col)
            .agg(F.count("*").alias("count"))
            .select("count")
            .rdd.map(lambda row: row.count)
            .collect()
        )
        
        if not key_counts:
            return {"skewed": False, "max": 0, "mean": 0, "ratio": 0}
        
        max_count = max(key_counts)
        mean_count = sum(key_counts) / len(key_counts)
        ratio = max_count / mean_count if mean_count > 0 else 0
        
        is_skewed = ratio > self.skew_threshold
        
        result = {
            "skewed": is_skewed,
            "max": max_count,
            "mean": mean_count,
            "ratio": ratio,
            "threshold": self.skew_threshold,
        }
        
        if is_skewed:
            self.logger.warning(
                f"Skew detected in '{key_col}': max/mean ratio = {ratio:.2f} "
                f"(threshold: {self.skew_threshold})"
            )
        
        return result
    
    def add_salting(
        self,
        df: SparkDataFrame,
        key_col: str,
        num_salts: int = 10,
        salt_col: Optional[str] = None,
    ) -> Tuple[SparkDataFrame, str]:
        """
        Add salt to a key column to mitigate skew.
        
        Args:
            df: DataFrame to salt
            key_col: Column to salt
            num_salts: Number of salt values
            salt_col: Name for salt column (auto-generated if None)
            
        Returns:
            Tuple of (salted DataFrame, salted key column name)
        """
        if salt_col is None:
            salt_col = f"{key_col}_salted"
        
        # Add random salt
        df_salted = df.withColumn(
            "salt",
            (F.rand(seed=42) * num_salts).cast("int")
        ).withColumn(
            salt_col,
            F.concat(F.col(key_col).cast("string"), F.lit("_"), F.col("salt").cast("string"))
        ).drop("salt")
        
        self.logger.info(
            f"Added salting to '{key_col}' with {num_salts} salts "
            f"(new column: '{salt_col}')"
        )
        
        return df_salted, salt_col
    
    def remove_salting(
        self,
        df: SparkDataFrame,
        salted_col: str,
        original_col: Optional[str] = None,
    ) -> SparkDataFrame:
        """
        Remove salt from a salted column.
        
        Args:
            df: DataFrame with salted column
            salted_col: Name of salted column
            original_col: Name for original column (uses salted_col without '_salted' if None)
            
        Returns:
            DataFrame with salt removed
        """
        if original_col is None:
            # Remove '_salted' suffix
            original_col = salted_col.replace("_salted", "")
        
        # Extract original key (remove salt suffix)
        df_clean = df.withColumn(
            original_col,
            F.split(F.col(salted_col), "_").getItem(0)
        ).drop(salted_col)
        
        self.logger.info(f"Removed salting from '{salted_col}' (restored to '{original_col}')")
        
        return df_clean
    
    def optimize_for_join(
        self,
        df: SparkDataFrame,
        join_key: str,
        num_partitions: Optional[int] = None,
    ) -> SparkDataFrame:
        """
        Optimize DataFrame for join operations.
        
        Args:
            df: DataFrame to optimize
            join_key: Key column for join
            num_partitions: Target partition count (auto-calculated if None)
            
        Returns:
            Repartitioned DataFrame
        """
        if num_partitions is None:
            # Use current shuffle partitions setting
            num_partitions = int(
                self.spark.conf.get("spark.sql.shuffle.partitions", "200")
            )
        
        current_partitions = df.rdd.getNumPartitions()
        
        if current_partitions != num_partitions:
            self.logger.info(
                f"Repartitioning for join on '{join_key}': "
                f"{current_partitions} -> {num_partitions} partitions"
            )
            return df.repartition(num_partitions, join_key)
        
        return df
    
    def _estimate_size(self, df: SparkDataFrame) -> int:
        """
        Estimate DataFrame size in bytes.
        
        Args:
            df: DataFrame to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Use Spark's size estimation
            rdd_id = df.rdd.id
            status = self.spark.sparkContext.statusTracker().getRDDInfo(rdd_id)
            if status:
                return status.memorySize + status.diskSize
        except Exception:
            pass
        
        # Fallback: rough estimate based on row count and schema
        try:
            num_rows = df.count()
            num_cols = len(df.columns)
            # Rough estimate: 100 bytes per row per column
            return num_rows * num_cols * 100
        except Exception:
            # Final fallback
            return 0
