"""
Spark implementations of feature validation.

Uses Spark aggregations for efficient distributed computation of validation metrics.
"""

from typing import Dict, List, Optional, Any, Set

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from cr_score.core.logging import get_audit_logger


class SparkFeatureValidator:
    """
    Spark-based feature validation utilities.
    
    Computes quality metrics using Spark aggregations for large datasets.
    """
    
    def __init__(
        self,
        warning_thresholds: Optional[Dict[str, float]] = None,
        hard_fail_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize validator.
        
        Args:
            warning_thresholds: Thresholds that trigger warnings
            hard_fail_thresholds: Thresholds that cause validation to fail
        """
        self.warning_thresholds = warning_thresholds or {}
        self.hard_fail_thresholds = hard_fail_thresholds or {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.logger = get_audit_logger()
    
    def validate_features(
        self,
        df: SparkDataFrame,
        feature_list: Optional[List[str]] = None,
        checks: Optional[List[str]] = None,
        sample_weight_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate features using Spark aggregations.
        
        Args:
            df: Input Spark DataFrame
            feature_list: Features to validate (None = all numeric columns)
            checks: Metrics to compute (None = all available)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            Dictionary with validation results per feature
        """
        available_checks = {
            'missing_rate', 'unique_count', 'zero_variance',
            'min', 'max', 'mean', 'std', 'p01', 'p99'
        }
        
        if checks is None:
            checks = list(available_checks)
        
        if feature_list is None:
            # Get numeric columns
            numeric_types = ['int', 'bigint', 'float', 'double', 'decimal']
            feature_list = [
                field.name for field in df.schema.fields
                if str(field.dataType).split('(')[0].lower() in numeric_types
            ]
        
        total_count = df.count()
        
        for feature in feature_list:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in DataFrame")
                continue
            
            result = {}
            
            # Basic metrics using Spark aggregations
            if 'missing_rate' in checks:
                if sample_weight_col:
                    null_count = df.filter(F.col(feature).isNull()).agg(
                        F.sum(sample_weight_col)
                    ).collect()[0][0] or 0.0
                    total_weight = df.agg(F.sum(sample_weight_col)).collect()[0][0]
                    result['missing_rate'] = null_count / total_weight if total_weight > 0 else 0.0
                else:
                    null_count = df.filter(F.col(feature).isNull()).count()
                    result['missing_rate'] = null_count / total_count if total_count > 0 else 0.0
            
            if 'unique_count' in checks:
                result['unique_count'] = df.select(feature).distinct().count()
            
            # Statistical metrics (only for numeric columns)
            numeric_checks = {'min', 'max', 'mean', 'std', 'p01', 'p99'}
            requested_numeric = set(checks) & numeric_checks
            
            if requested_numeric:
                # Compute all numeric metrics in one pass
                agg_exprs = []
                if 'min' in requested_numeric:
                    agg_exprs.append(F.min(feature).alias('min'))
                if 'max' in requested_numeric:
                    agg_exprs.append(F.max(feature).alias('max'))
                if 'mean' in requested_numeric:
                    agg_exprs.append(F.avg(feature).alias('mean'))
                if 'std' in requested_numeric:
                    agg_exprs.append(F.stddev(feature).alias('std'))
                if 'p01' in requested_numeric or 'p99' in requested_numeric:
                    # Use percentile_approx for percentiles
                    if 'p01' in requested_numeric:
                        agg_exprs.append(F.expr(f'percentile_approx({feature}, 0.01)').alias('p01'))
                    if 'p99' in requested_numeric:
                        agg_exprs.append(F.expr(f'percentile_approx({feature}, 0.99)').alias('p99'))
                
                if agg_exprs:
                    stats = df.agg(*agg_exprs).collect()[0]
                    for check in requested_numeric:
                        if check in stats.asDict():
                            result[check] = stats[check]
            
            # Zero variance check
            if 'zero_variance' in checks:
                if 'std' in result:
                    result['zero_variance'] = result['std'] == 0 or result['std'] is None
                else:
                    # Compute std just for this check
                    std_val = df.agg(F.stddev(feature)).collect()[0][0]
                    result['zero_variance'] = std_val == 0 or std_val is None
            
            # Check thresholds
            warnings_list = []
            failures = []
            
            for metric, value in result.items():
                if value is None:
                    continue
                    
                if metric in self.hard_fail_thresholds:
                    if value > self.hard_fail_thresholds[metric]:
                        failures.append(
                            f"{metric}={value:.4f} exceeds hard threshold {self.hard_fail_thresholds[metric]}"
                        )
                
                if metric in self.warning_thresholds:
                    if value > self.warning_thresholds[metric]:
                        warnings_list.append(
                            f"{metric}={value:.4f} exceeds warning threshold {self.warning_thresholds[metric]}"
                        )
            
            result['warnings'] = warnings_list
            result['failures'] = failures
            result['status'] = 'FAIL' if failures else ('WARN' if warnings_list else 'PASS')
            
            self.results[feature] = result
        
        return self.results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert validation results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for feature, metrics in self.results.items():
            row = {'feature': feature}
            row.update({k: v for k, v in metrics.items() if k not in ['warnings', 'failures']})
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export_csv(self, path: str) -> None:
        """Export validation results to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        self.logger.info(f"Exported validation results to {path}")
    
    def export_json(self, path: str) -> None:
        """Export validation results to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.info(f"Exported validation results to {path}")
    
    def compute_psi(
        self,
        baseline_df: SparkDataFrame,
        current_df: SparkDataFrame,
        feature_col: str,
        bins: int = 10,
        sample_weight_col: Optional[str] = None,
    ) -> float:
        """
        Compute Population Stability Index (PSI) using Spark aggregations.
        
        Args:
            baseline_df: Baseline Spark DataFrame
            current_df: Current Spark DataFrame
            feature_col: Feature column to compute PSI for
            bins: Number of bins for discretization
            sample_weight_col: Optional sample weight column
            
        Returns:
            PSI value
        """
        # Get min/max from baseline
        baseline_stats = baseline_df.agg(
            F.min(feature_col).alias('min'),
            F.max(feature_col).alias('max')
        ).collect()[0]
        
        min_val = baseline_stats['min']
        max_val = baseline_stats['max']
        
        if min_val is None or max_val is None:
            return 0.0
        
        # Create bin edges
        bin_width = (max_val - min_val) / bins
        bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
        
        # Bin baseline distribution
        baseline_binned = baseline_df.withColumn(
            'bin',
            F.floor((F.col(feature_col) - F.lit(min_val)) / F.lit(bin_width))
            .cast('int')
        )
        
        if sample_weight_col:
            baseline_total = baseline_df.agg(F.sum(sample_weight_col)).collect()[0][0] or 1.0
            baseline_props = (
                baseline_binned
                .groupBy('bin')
                .agg(F.sum(sample_weight_col).alias('weight'))
                .withColumn('prop', F.col('weight') / F.lit(baseline_total))
                .select('bin', 'prop')
            )
        else:
            baseline_count = baseline_df.count()
            baseline_props = (
                baseline_binned
                .groupBy('bin')
                .agg(F.count('*').alias('count'))
                .withColumn('prop', F.col('count') / F.lit(baseline_count))
                .select('bin', 'prop')
            )
        
        # Bin current distribution
        current_binned = current_df.withColumn(
            'bin',
            F.floor((F.col(feature_col) - F.lit(min_val)) / F.lit(bin_width))
            .cast('int')
        )
        
        if sample_weight_col:
            current_total = current_df.agg(F.sum(sample_weight_col)).collect()[0][0] or 1.0
            current_props = (
                current_binned
                .groupBy('bin')
                .agg(F.sum(sample_weight_col).alias('weight'))
                .withColumn('prop', F.col('weight') / F.lit(current_total))
                .select('bin', 'prop')
            )
        else:
            current_count = current_df.count()
            current_props = (
                current_binned
                .groupBy('bin')
                .agg(F.count('*').alias('count'))
                .withColumn('prop', F.col('count') / F.lit(current_count))
                .select('bin', 'prop')
            )
        
        # Join and compute PSI (rename columns to avoid conflicts)
        baseline_props_renamed = baseline_props.select('bin', F.col('prop').alias('baseline_prop'))
        current_props_renamed = current_props.select('bin', F.col('prop').alias('current_prop'))
        
        psi_df = (
            baseline_props_renamed
            .join(current_props_renamed, 'bin', 'full_outer')
            .withColumn('baseline_prop', F.coalesce('baseline_prop', F.lit(0.0001)))
            .withColumn('current_prop', F.coalesce('current_prop', F.lit(0.0001)))
            .withColumn(
                'psi_contrib',
                (F.col('current_prop') - F.col('baseline_prop')) *
                F.log(F.col('current_prop') / F.col('baseline_prop'))
            )
        )
        
        psi = psi_df.agg(F.sum('psi_contrib')).collect()[0][0] or 0.0
        
        return float(psi)
