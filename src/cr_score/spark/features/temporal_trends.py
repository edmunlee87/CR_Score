"""
Spark implementations of temporal trend features.

Uses Spark Window functions for efficient distributed processing of large datasets.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Window, functions as F
from pyspark.sql.types import DoubleType

from cr_score.core.logging import get_audit_logger


class SparkTemporalTrendFeatures:
    """
    Spark-based temporal trend feature generators.
    
    Optimized for distributed processing using Spark Window functions.
    """
    
    def __init__(self) -> None:
        """Initialize Spark temporal feature generator."""
        self.logger = get_audit_logger()
    
    def delta(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        periods: int = 1,
    ) -> SparkDataFrame:
        """
        Compute delta (difference from previous period) using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute delta on
            output_col: Output column name (default: {column}_delta)
            time_col: Time column for sorting
            group_cols: Columns to group by
            periods: Number of periods to look back
            
        Returns:
            Spark DataFrame with delta column
        """
        if output_col is None:
            output_col = f"{column}_delta"
        
        # Build window specification
        if group_cols and time_col:
            window = Window.partitionBy(*group_cols).orderBy(time_col)
        elif group_cols:
            window = Window.partitionBy(*group_cols)
        elif time_col:
            window = Window.orderBy(time_col)
        else:
            window = Window.rowsBetween(-periods, 0)
        
        # Compute delta using lag
        df = df.withColumn(
            output_col,
            F.col(column) - F.lag(F.col(column), periods).over(window)
        )
        
        return df
    
    def pct_change(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        periods: int = 1,
        fill_method: Optional[str] = None,
    ) -> SparkDataFrame:
        """
        Compute percent change from previous period using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute percent change on
            output_col: Output column name (default: {column}_pct_change)
            time_col: Time column for sorting
            group_cols: Columns to group by
            periods: Number of periods to look back
            fill_method: Method to fill NaN values (not used in Spark, kept for API compatibility)
            
        Returns:
            Spark DataFrame with percent change column
        """
        if output_col is None:
            output_col = f"{column}_pct_change"
        
        # Build window specification
        if group_cols and time_col:
            window = Window.partitionBy(*group_cols).orderBy(time_col)
        elif group_cols:
            window = Window.partitionBy(*group_cols)
        elif time_col:
            window = Window.orderBy(time_col)
        else:
            window = Window.rowsBetween(-periods, 0)
        
        # Compute percent change: (current - previous) / previous
        lag_value = F.lag(F.col(column), periods).over(window)
        df = df.withColumn(
            output_col,
            (F.col(column) - lag_value) / F.when(lag_value != 0, lag_value).otherwise(1.0)
        )
        
        return df
    
    def momentum(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 3,
    ) -> SparkDataFrame:
        """
        Compute momentum (current value - rolling mean) using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute momentum on
            output_col: Output column name (default: {column}_momentum)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            Spark DataFrame with momentum column
        """
        if output_col is None:
            output_col = f"{column}_momentum_{window}"
        
        # Build window specification
        if group_cols and time_col:
            window_spec = Window.partitionBy(*group_cols).orderBy(time_col).rowsBetween(-window + 1, 0)
        elif group_cols:
            window_spec = Window.partitionBy(*group_cols).rowsBetween(-window + 1, 0)
        elif time_col:
            window_spec = Window.orderBy(time_col).rowsBetween(-window + 1, 0)
        else:
            window_spec = Window.rowsBetween(-window + 1, 0)
        
        # Compute rolling mean and momentum
        rolling_mean = F.avg(F.col(column)).over(window_spec)
        df = df.withColumn(output_col, F.col(column) - rolling_mean)
        
        return df
    
    def volatility(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
        method: str = "std",
    ) -> SparkDataFrame:
        """
        Compute volatility (rolling std or CV) using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute volatility on
            output_col: Output column name (default: {column}_volatility)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            method: "std" for standard deviation or "cv" for coefficient of variation
            
        Returns:
            Spark DataFrame with volatility column
        """
        if output_col is None:
            output_col = f"{column}_volatility_{window}"
        
        # Build window specification
        if group_cols and time_col:
            window_spec = Window.partitionBy(*group_cols).orderBy(time_col).rowsBetween(-window + 1, 0)
        elif group_cols:
            window_spec = Window.partitionBy(*group_cols).rowsBetween(-window + 1, 0)
        elif time_col:
            window_spec = Window.orderBy(time_col).rowsBetween(-window + 1, 0)
        else:
            window_spec = Window.rowsBetween(-window + 1, 0)
        
        # Compute volatility
        if method == "std":
            df = df.withColumn(output_col, F.stddev(F.col(column)).over(window_spec))
        elif method == "cv":
            rolling_mean = F.avg(F.col(column)).over(window_spec)
            rolling_std = F.stddev(F.col(column)).over(window_spec)
            df = df.withColumn(
                output_col,
                rolling_std / (rolling_mean + F.lit(1e-8))
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'std' or 'cv'.")
        
        return df
    
    def trend_slope(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
    ) -> SparkDataFrame:
        """
        Compute trend slope using linear regression over rolling window.
        
        Note: This is a simplified implementation. For exact linear regression,
        consider using Spark ML or collecting small windows to driver.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute trend slope on
            output_col: Output column name (default: {column}_trend_slope)
            time_col: Time column for sorting (required for trend slope)
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            Spark DataFrame with trend slope column
        """
        if output_col is None:
            output_col = f"{column}_trend_slope_{window}"
        
        if time_col is None:
            raise ValueError("time_col is required for trend_slope calculation")
        
        # For Spark, we'll use a simplified approach: (last - first) / window
        # For exact linear regression, would need UDF or Spark ML
        if group_cols:
            window_spec = Window.partitionBy(*group_cols).orderBy(time_col).rowsBetween(-window + 1, 0)
        else:
            window_spec = Window.orderBy(time_col).rowsBetween(-window + 1, 0)
        
        # Simplified trend: (last - first) / (window - 1)
        first_value = F.first(F.col(column)).over(window_spec)
        last_value = F.last(F.col(column)).over(window_spec)
        df = df.withColumn(
            output_col,
            (last_value - first_value) / F.lit(max(window - 1, 1))
        )
        
        return df
    
    def rolling_rank(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
    ) -> SparkDataFrame:
        """
        Compute rolling rank within window using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute rank on
            output_col: Output column name (default: {column}_rolling_rank)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            Spark DataFrame with rolling rank column
        """
        if output_col is None:
            output_col = f"{column}_rolling_rank_{window}"
        
        # Build window specification
        if group_cols and time_col:
            window_spec = Window.partitionBy(*group_cols).orderBy(time_col).rowsBetween(-window + 1, 0)
        elif group_cols:
            window_spec = Window.partitionBy(*group_cols).rowsBetween(-window + 1, 0)
        elif time_col:
            window_spec = Window.orderBy(time_col).rowsBetween(-window + 1, 0)
        else:
            window_spec = Window.rowsBetween(-window + 1, 0)
        
        # Compute rank within window
        df = df.withColumn(output_col, F.rank().over(window_spec.orderBy(F.col(column).desc())))
        
        return df
    
    def minmax_range(
        self,
        df: SparkDataFrame,
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
    ) -> SparkDataFrame:
        """
        Compute min-max range over rolling window using Spark Window functions.
        
        Args:
            df: Input Spark DataFrame
            column: Column to compute range on
            output_col: Output column name (default: {column}_minmax_range)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            Spark DataFrame with minmax range column
        """
        if output_col is None:
            output_col = f"{column}_minmax_range_{window}"
        
        # Build window specification
        if group_cols and time_col:
            window_spec = Window.partitionBy(*group_cols).orderBy(time_col).rowsBetween(-window + 1, 0)
        elif group_cols:
            window_spec = Window.partitionBy(*group_cols).rowsBetween(-window + 1, 0)
        elif time_col:
            window_spec = Window.orderBy(time_col).rowsBetween(-window + 1, 0)
        else:
            window_spec = Window.rowsBetween(-window + 1, 0)
        
        # Compute range: max - min
        rolling_max = F.max(F.col(column)).over(window_spec)
        rolling_min = F.min(F.col(column)).over(window_spec)
        df = df.withColumn(output_col, rolling_max - rolling_min)
        
        return df
