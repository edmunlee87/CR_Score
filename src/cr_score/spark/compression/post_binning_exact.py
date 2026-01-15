"""
Post-binning exact compression with sample weighting.

Implements URD Appendix D: Replace N identical rows with 1 row + weight,
preserving likelihoods and event rates exactly.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cr_score.core.exceptions import CompressionError
from cr_score.core.logging import get_audit_logger


class PostBinningCompressor:
    """
    Exact compression of binned data using sample weighting.

    After binning, rows with identical bin assignments can be aggregated
    without loss of information. This reduces data volume by 20x-100x.

    Example:
        >>> compressor = PostBinningCompressor(spark)
        >>> compressed_df = compressor.compress(
        ...     binned_df,
        ...     bin_columns=["bin_age", "bin_income", "bin_credit_score"],
        ...     target_col="target",
        ...     segment_cols=["product", "channel"]
        ... )
        >>> print(f"Compression ratio: {compressor.get_compression_ratio():.1f}x")
    """

    def __init__(self, spark: SparkSession) -> None:
        """
        Initialize compressor.

        Args:
            spark: Active Spark session
        """
        self.spark = spark
        self.logger = get_audit_logger()
        self.original_count: Optional[int] = None
        self.compressed_count: Optional[int] = None

    def compress(
        self,
        df: SparkDataFrame,
        bin_columns: List[str],
        target_col: str,
        segment_cols: Optional[List[str]] = None,
        verify: bool = True,
    ) -> SparkDataFrame:
        """
        Compress DataFrame by aggregating identical binned rows.

        Args:
            df: Binned DataFrame
            bin_columns: Columns containing bin assignments
            target_col: Target variable column
            segment_cols: Additional grouping columns (product, channel, etc.)
            verify: If True, verify compression preserves totals

        Returns:
            Compressed DataFrame with sample_weight and event_weight columns

        Raises:
            CompressionError: If verification fails

        Example:
            >>> compressed = compressor.compress(
            ...     df,
            ...     bin_columns=["bin_age", "bin_income"],
            ...     target_col="default",
            ...     segment_cols=["product"]
            ... )
        """
        self.logger.info(
            "Starting post-binning compression",
            bin_columns=bin_columns,
            target_col=target_col,
            segment_cols=segment_cols,
        )

        # Store original count
        self.original_count = df.count()

        # Build group-by columns
        group_cols = bin_columns.copy()
        if segment_cols:
            group_cols.extend(segment_cols)

        # Aggregate
        compressed = (
            df.groupBy(group_cols)
            .agg(
                F.count("*").alias("sample_weight"),
                F.sum(target_col).alias("event_weight"),
            )
            .withColumn(
                "event_rate",
                F.col("event_weight") / F.col("sample_weight")
            )
        )

        # Store compressed count
        self.compressed_count = compressed.count()

        self.logger.info(
            "Compression completed",
            original_rows=self.original_count,
            compressed_rows=self.compressed_count,
            compression_ratio=self.get_compression_ratio(),
        )

        # Verification
        if verify:
            self._verify_compression(df, compressed, target_col)

        return compressed

    def compress_pandas(
        self,
        df: pd.DataFrame,
        bin_columns: List[str],
        target_col: str,
        segment_cols: Optional[List[str]] = None,
        verify: bool = True,
    ) -> pd.DataFrame:
        """
        Compress pandas DataFrame (for python_local engine).

        Args:
            df: Binned DataFrame
            bin_columns: Columns containing bin assignments
            target_col: Target variable column
            segment_cols: Additional grouping columns
            verify: If True, verify compression preserves totals

        Returns:
            Compressed DataFrame with sample_weight and event_weight columns

        Raises:
            CompressionError: If verification fails

        Example:
            >>> compressed = compressor.compress_pandas(
            ...     df,
            ...     bin_columns=["bin_age", "bin_income"],
            ...     target_col="default"
            ... )
        """
        self.logger.info(
            "Starting post-binning compression (pandas)",
            bin_columns=bin_columns,
            target_col=target_col,
        )

        self.original_count = len(df)

        # Build group-by columns
        group_cols = bin_columns.copy()
        if segment_cols:
            group_cols.extend(segment_cols)

        # Aggregate
        compressed = (
            df.groupby(group_cols, as_index=False, dropna=False)
            .agg(
                sample_weight=("__index__" if "__index__" in df.columns else df.columns[0], "size"),
                event_weight=(target_col, "sum"),
            )
        )

        # Compute event rate
        compressed["event_rate"] = compressed["event_weight"] / compressed["sample_weight"]

        self.compressed_count = len(compressed)

        self.logger.info(
            "Compression completed (pandas)",
            original_rows=self.original_count,
            compressed_rows=self.compressed_count,
            compression_ratio=self.get_compression_ratio(),
        )

        # Verification
        if verify:
            self._verify_compression_pandas(df, compressed, target_col)

        return compressed

    def _verify_compression(
        self,
        original: SparkDataFrame,
        compressed: SparkDataFrame,
        target_col: str,
        tolerance: float = 0.0,
    ) -> None:
        """
        Verify compression preserves totals exactly.

        Args:
            original: Original DataFrame
            compressed: Compressed DataFrame
            target_col: Target column
            tolerance: Allowed relative difference (default 0.0 for exact)

        Raises:
            CompressionError: If totals don't match within tolerance
        """
        # Check total observations
        original_total = original.count()
        compressed_total = compressed.agg(F.sum("sample_weight")).collect()[0][0]

        if original_total != compressed_total:
            raise CompressionError(
                "Sample weight sum does not match original row count",
                details={
                    "original_count": original_total,
                    "compressed_weight_sum": compressed_total,
                    "difference": compressed_total - original_total,
                }
            )

        # Check total events
        original_events = original.agg(F.sum(target_col)).collect()[0][0]
        compressed_events = compressed.agg(F.sum("event_weight")).collect()[0][0]

        rel_diff = abs(compressed_events - original_events) / original_events if original_events > 0 else 0

        if rel_diff > tolerance:
            raise CompressionError(
                "Event weight sum does not match original event count",
                details={
                    "original_events": original_events,
                    "compressed_events": compressed_events,
                    "relative_difference": rel_diff,
                    "tolerance": tolerance,
                }
            )

        self.logger.info(
            "Compression verification passed",
            original_count=original_total,
            original_events=original_events,
            relative_difference=rel_diff,
        )

    def _verify_compression_pandas(
        self,
        original: pd.DataFrame,
        compressed: pd.DataFrame,
        target_col: str,
        tolerance: float = 0.0,
    ) -> None:
        """
        Verify pandas compression preserves totals exactly.

        Args:
            original: Original DataFrame
            compressed: Compressed DataFrame
            target_col: Target column
            tolerance: Allowed relative difference

        Raises:
            CompressionError: If totals don't match within tolerance
        """
        # Check total observations
        original_total = len(original)
        compressed_total = compressed["sample_weight"].sum()

        if original_total != compressed_total:
            raise CompressionError(
                "Sample weight sum does not match original row count",
                details={
                    "original_count": original_total,
                    "compressed_weight_sum": int(compressed_total),
                    "difference": int(compressed_total - original_total),
                }
            )

        # Check total events
        original_events = original[target_col].sum()
        compressed_events = compressed["event_weight"].sum()

        rel_diff = abs(compressed_events - original_events) / original_events if original_events > 0 else 0

        if rel_diff > tolerance:
            raise CompressionError(
                "Event weight sum does not match original event count",
                details={
                    "original_events": float(original_events),
                    "compressed_events": float(compressed_events),
                    "relative_difference": float(rel_diff),
                    "tolerance": tolerance,
                }
            )

        self.logger.info(
            "Compression verification passed (pandas)",
            original_count=original_total,
            original_events=float(original_events),
            relative_difference=float(rel_diff),
        )

    def get_compression_ratio(self) -> float:
        """
        Get compression ratio (original / compressed).

        Returns:
            Compression ratio (e.g., 50.0 means 50x compression)

        Example:
            >>> ratio = compressor.get_compression_ratio()
            >>> print(f"Compressed by {ratio:.1f}x")
        """
        if self.original_count is None or self.compressed_count is None:
            return 1.0
        return self.original_count / self.compressed_count if self.compressed_count > 0 else 1.0

    def get_memory_savings_pct(self) -> float:
        """
        Get memory savings percentage.

        Returns:
            Percentage of memory saved

        Example:
            >>> savings = compressor.get_memory_savings_pct()
            >>> print(f"Memory saved: {savings:.1f}%")
        """
        ratio = self.get_compression_ratio()
        return (1 - 1/ratio) * 100 if ratio > 1 else 0.0

    def generate_compression_report(self) -> Dict[str, Any]:
        """
        Generate compression summary report.

        Returns:
            Dictionary with compression statistics

        Example:
            >>> report = compressor.generate_compression_report()
            >>> print(report)
        """
        return {
            "original_rows": self.original_count,
            "compressed_rows": self.compressed_count,
            "compression_ratio": self.get_compression_ratio(),
            "memory_savings_pct": self.get_memory_savings_pct(),
            "rows_eliminated": self.original_count - self.compressed_count if self.original_count and self.compressed_count else 0,
        }
