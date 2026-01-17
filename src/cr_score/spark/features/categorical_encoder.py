"""
Spark implementations of categorical encoding.

Uses broadcast joins for efficient distributed processing of large datasets.
"""

from typing import Dict, List, Optional, Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from cr_score.core.logging import get_audit_logger


class SparkCategoricalEncoder:
    """
    Spark-based categorical encoding utilities.
    
    Optimized for distributed processing using broadcast joins.
    """
    
    def __init__(self, handle_missing: str = "category") -> None:
        """
        Initialize encoder.
        
        Args:
            handle_missing: How to handle missing values ("category", "mode", "drop")
        """
        self.handle_missing = handle_missing
        self.logger = get_audit_logger()
        self.mappings: Dict[str, Dict[str, Any]] = {}
    
    def freq_encoding(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        sample_weight_col: Optional[str] = None,
    ) -> SparkDataFrame:
        """
        Encode categorical variable with frequency using Spark aggregations.
        
        Args:
            df: Input Spark DataFrame
            column: Column to encode
            output_col: Output column name (default: {column}_freq)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            Spark DataFrame with encoded column
        """
        if output_col is None:
            output_col = f"{column}_freq"
        
        # Compute frequency map
        if sample_weight_col:
            # Weighted frequency
            total_weight = df.agg(F.sum(sample_weight_col)).collect()[0][0]
            freq_map = (
                df.groupBy(column)
                .agg(F.sum(sample_weight_col).alias("weight"))
                .withColumn("freq", F.col("weight") / F.lit(total_weight))
                .select(column, "freq")
            )
        else:
            # Standard frequency
            total_count = df.count()
            freq_map = (
                df.groupBy(column)
                .agg(F.count("*").alias("count"))
                .withColumn("freq", F.col("count") / F.lit(total_count))
                .select(column, "freq")
            )
        
        # Broadcast join for efficiency
        df = df.join(broadcast(freq_map), column, "left")
        
        # Rename and handle unseen categories
        df = df.withColumn(
            output_col,
            F.coalesce(F.col("freq"), F.lit(0.0))
        ).drop("freq")
        
        # Store mapping (collect small lookup table)
        mapping_dict = {row[column]: row["freq"] for row in freq_map.collect()}
        self.mappings[output_col] = {"type": "frequency", "mapping": mapping_dict}
        
        return df
    
    def target_mean_encoding(
        self,
        df: SparkDataFrame,
        column: str,
        target: str,
        output_col: Optional[str] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        sample_weight_col: Optional[str] = None,
    ) -> SparkDataFrame:
        """
        Target mean encoding with smoothing using Spark aggregations.
        
        Args:
            df: Input Spark DataFrame
            column: Column to encode
            target: Target column
            output_col: Output column name (default: {column}_target_mean)
            smoothing: Smoothing parameter
            min_samples_leaf: Minimum samples per category
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            Spark DataFrame with encoded column
        """
        if output_col is None:
            output_col = f"{column}_target_mean"
        
        # Compute global mean
        if sample_weight_col:
            global_mean = (
                df.agg(
                    F.sum(F.col(target) * F.col(sample_weight_col)) / F.sum(sample_weight_col)
                ).collect()[0][0]
            )
        else:
            global_mean = df.agg(F.avg(target)).collect()[0][0]
        
        # Compute category statistics
        if sample_weight_col:
            cat_stats = (
                df.groupBy(column)
                .agg(
                    F.sum(F.col(target) * F.col(sample_weight_col)).alias("weighted_sum"),
                    F.sum(sample_weight_col).alias("weight")
                )
                .withColumn("mean", F.col("weighted_sum") / F.col("weight"))
                .withColumn("count", F.col("weight"))
            )
        else:
            cat_stats = (
                df.groupBy(column)
                .agg(
                    F.avg(target).alias("mean"),
                    F.count("*").alias("count")
                )
            )
        
        # Smoothed mean
        smoothed_map = (
            cat_stats
            .filter(F.col("count") >= min_samples_leaf)
            .withColumn(
                "smoothed_mean",
                (F.col("count") * F.col("mean") + F.lit(smoothing) * F.lit(global_mean)) /
                (F.col("count") + F.lit(smoothing))
            )
            .select(column, "smoothed_mean")
        )
        
        # Broadcast join
        df = df.join(broadcast(smoothed_map), column, "left")
        
        # Rename and handle unseen categories
        df = df.withColumn(
            output_col,
            F.coalesce(F.col("smoothed_mean"), F.lit(global_mean))
        ).drop("smoothed_mean")
        
        # Store mapping
        mapping_dict = {row[column]: row["smoothed_mean"] for row in smoothed_map.collect()}
        self.mappings[output_col] = {
            "type": "target_mean",
            "mapping": mapping_dict,
            "global_mean": global_mean,
            "smoothing": smoothing,
        }
        
        return df
    
    def rare_grouping(
        self,
        df: SparkDataFrame,
        column: str,
        threshold: float = 0.01,
        rare_label: str = "RARE",
        output_col: Optional[str] = None,
        sample_weight_col: Optional[str] = None,
    ) -> SparkDataFrame:
        """
        Group rare categories together using Spark aggregations.
        
        Args:
            df: Input Spark DataFrame
            column: Column to process
            threshold: Frequency threshold for rare categories
            rare_label: Label for rare categories
            output_col: Output column name (default: {column}_grouped)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            Spark DataFrame with grouped column
        """
        if output_col is None:
            output_col = f"{column}_grouped"
        
        # Compute frequencies
        if sample_weight_col:
            total_weight = df.agg(F.sum(sample_weight_col)).collect()[0][0]
            freq_map = (
                df.groupBy(column)
                .agg(F.sum(sample_weight_col).alias("weight"))
                .withColumn("freq", F.col("weight") / F.lit(total_weight))
                .select(column, "freq")
            )
        else:
            total_count = df.count()
            freq_map = (
                df.groupBy(column)
                .agg(F.count("*").alias("count"))
                .withColumn("freq", F.col("count") / F.lit(total_count))
                .select(column, "freq")
            )
        
        # Identify rare categories
        rare_cats = (
            freq_map
            .filter(F.col("freq") < threshold)
            .select(column)
        )
        
        # Create mapping: rare -> RARE, others -> original
        rare_map = (
            rare_cats
            .withColumn("grouped", F.lit(rare_label))
            .select(column, "grouped")
        )
        
        # Broadcast join and coalesce
        df = df.join(broadcast(rare_map), column, "left")
        df = df.withColumn(
            output_col,
            F.coalesce(F.col("grouped"), F.col(column))
        ).drop("grouped")
        
        # Store mapping
        rare_list = [row[column] for row in rare_cats.collect()]
        self.mappings[output_col] = {
            "type": "rare_grouping",
            "rare_categories": rare_list,
            "threshold": threshold,
        }
        
        return df
    
    def export_mappings(self, path: str) -> None:
        """Export encodings to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(self.mappings, f, indent=2, default=str)
        self.logger.info(f"Exported encoding mappings to {path}")
