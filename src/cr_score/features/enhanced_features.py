"""
Enhanced feature engineering utilities.

Additional tools for:
- Temporal trend features
- Categorical encoding
- Feature validation
- Dependency graph management
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from cr_score.core.logging import get_audit_logger

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import Window, functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None


class DependencyGraph:
    """
    Manage feature dependencies and determine execution order.
    
    Builds a directed graph of feature dependencies and performs
    topological sort to determine the correct order of execution.
    """
    
    def __init__(self) -> None:
        """Initialize dependency graph."""
        self.graph: Dict[str, List[str]] = {}
        self.logger = get_audit_logger()
    
    def add_feature(self, feature: str, dependencies: List[str]) -> None:
        """Add a feature and its dependencies to the graph."""
        if feature not in self.graph:
            self.graph[feature] = []
        
        for dep in dependencies:
            if dep not in self.graph:
                self.graph[dep] = []
            self.graph[feature].append(dep)
    
    def detect_cycle(self) -> Optional[List[str]]:
        """
        Detect cycles in the dependency graph.
        
        Returns:
            List of features in the cycle, or None if no cycle exists
        """
        visited = set()
        rec_stack = set()
        
        def visit(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    cycle = visit(neighbor, path.copy())
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        for node in self.graph:
            if node not in visited:
                cycle = visit(node, [])
                if cycle:
                    return cycle
        
        return None
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort on the dependency graph.
        
        Returns:
            List of features in execution order
            
        Raises:
            ValueError: If a cycle is detected
        """
        # Check for cycles
        cycle = self.detect_cycle()
        if cycle:
            raise ValueError(f"Dependency cycle detected: {' -> '.join(cycle)}")
        
        # Perform topological sort (Kahn's algorithm)
        # Note: In our graph, edges go FROM dependent TO dependency
        # So we need to reverse the logic
        
        in_degree = {node: 0 for node in self.graph}
        
        # Count incoming edges (features that depend on this node)
        for node in self.graph:
            for neighbor in self.graph[node]:
                if neighbor in in_degree:
                    in_degree[node] += 1
        
        # Start with nodes that have no dependencies
        queue = [node for node in self.graph if len(self.graph[node]) == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Find nodes that depend on this node
            for other_node in self.graph:
                if node in self.graph[other_node]:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        if len(result) != len(self.graph):
            raise ValueError("Graph has a cycle")
        
        return result
    
    def get_dependencies(self, feature: str) -> Set[str]:
        """Get all dependencies (direct and transitive) for a feature."""
        deps = set()
        
        def collect_deps(feat: str) -> None:
            for dep in self.graph.get(feat, []):
                if dep not in deps:
                    deps.add(dep)
                    collect_deps(dep)
        
        collect_deps(feature)
        return deps


class FeatureValidator:
    """
    Validate features and generate quality metrics.
    
    Computes various statistics and checks for data quality issues
    such as missing values, zero variance, outliers, etc.
    """
    
    def __init__(
        self,
        hard_fail_thresholds: Optional[Dict[str, float]] = None,
        warning_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize validator.
        
        Args:
            hard_fail_thresholds: Thresholds that cause validation to fail
            warning_thresholds: Thresholds that generate warnings
        """
        self.hard_fail_thresholds = hard_fail_thresholds or {}
        self.warning_thresholds = warning_thresholds or {}
        self.logger = get_audit_logger()
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def validate_features(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        feature_list: Optional[List[str]] = None,
        checks: Optional[List[str]] = None,
        sample_weight_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate features and compute quality metrics.
        
        Args:
            df: DataFrame containing features (pandas or Spark)
            feature_list: List of features to validate (all numeric if None)
            checks: List of checks to perform (all if None)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            Dictionary of validation results per feature
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.feature_validator import SparkFeatureValidator
            spark_validator = SparkFeatureValidator(
                self.warning_thresholds, self.hard_fail_thresholds
            )
            result = spark_validator.validate_features(df, feature_list, checks, sample_weight_col)
            self.results.update(result)
            return result
        
        # Pandas implementation
        if feature_list is None:
            feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        available_checks = {
            'missing_rate', 'unique_count', 'zero_variance',
            'min', 'max', 'mean', 'std', 'p01', 'p99',
            'skewness', 'kurtosis'
        }
        
        if checks is None:
            checks = list(available_checks)
        
        for feature in feature_list:
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in DataFrame")
                continue
            
            result = {}
            series = df[feature]
            
            # Basic metrics
            if 'missing_rate' in checks:
                result['missing_rate'] = series.isna().mean()
            
            if 'unique_count' in checks:
                result['unique_count'] = series.nunique()
            
            if 'zero_variance' in checks:
                result['zero_variance'] = series.std() == 0 if series.notna().any() else True
            
            # Statistical metrics
            if series.dtype in [np.number, 'float64', 'int64']:
                if 'min' in checks:
                    result['min'] = series.min()
                if 'max' in checks:
                    result['max'] = series.max()
                if 'mean' in checks:
                    result['mean'] = series.mean()
                if 'std' in checks:
                    result['std'] = series.std()
                if 'p01' in checks:
                    result['p01'] = series.quantile(0.01)
                if 'p99' in checks:
                    result['p99'] = series.quantile(0.99)
                if 'skewness' in checks:
                    result['skewness'] = series.skew()
                if 'kurtosis' in checks:
                    result['kurtosis'] = series.kurtosis()
            
            # Check thresholds
            warnings_list = []
            failures = []
            
            for metric, value in result.items():
                if metric in self.hard_fail_thresholds:
                    if value > self.hard_fail_thresholds[metric]:
                        failures.append(f"{metric}={value:.4f} exceeds hard threshold {self.hard_fail_thresholds[metric]}")
                
                if metric in self.warning_thresholds:
                    if value > self.warning_thresholds[metric]:
                        warnings_list.append(f"{metric}={value:.4f} exceeds warning threshold {self.warning_thresholds[metric]}")
            
            result['warnings'] = warnings_list
            result['failures'] = failures
            result['status'] = 'FAIL' if failures else ('WARN' if warnings_list else 'PASS')
            
            self.results[feature] = result
        
        return self.results
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert validation results to DataFrame."""
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
        baseline_dist: Union[pd.Series, "SparkDataFrame"],
        current_dist: Union[pd.Series, "SparkDataFrame"],
        bins: int = 10,
        feature_col: Optional[str] = None,
        sample_weight_col: Optional[str] = None,
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        Args:
            baseline_dist: Baseline distribution (pandas Series or Spark DataFrame)
            current_dist: Current distribution (pandas Series or Spark DataFrame)
            bins: Number of bins for discretization
            feature_col: Feature column name (required if Spark DataFrame)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            PSI value
        """
        # Auto-detect Spark DataFrame
        if isinstance(baseline_dist, SparkDataFrame) or isinstance(current_dist, SparkDataFrame):
            if feature_col is None:
                raise ValueError("feature_col is required when using Spark DataFrames")
            from cr_score.spark.features.feature_validator import SparkFeatureValidator
            spark_validator = SparkFeatureValidator()
            return spark_validator.compute_psi(
                baseline_dist, current_dist, feature_col, bins, sample_weight_col
            )
        
        # Pandas implementation
        # Create bins from baseline
        _, bin_edges = np.histogram(baseline_dist.dropna(), bins=bins)
        
        # Bin both distributions
        baseline_binned = np.digitize(baseline_dist, bin_edges)
        current_binned = np.digitize(current_dist, bin_edges)
        
        # Compute proportions
        baseline_props = pd.Series(baseline_binned).value_counts(normalize=True)
        current_props = pd.Series(current_binned).value_counts(normalize=True)
        
        # Align indices
        all_bins = set(baseline_props.index) | set(current_props.index)
        baseline_props = baseline_props.reindex(all_bins, fill_value=0.0001)
        current_props = current_props.reindex(all_bins, fill_value=0.0001)
        
        # Compute PSI
        psi = ((current_props - baseline_props) * np.log(current_props / baseline_props)).sum()
        
        return psi


class TemporalTrendFeatures:
    """
    Temporal trend feature generators for time series analysis.
    
    Provides delta, percent change, momentum, volatility, trend slope,
    rolling rank, and other temporal features critical for credit risk.
    
    Automatically detects Spark DataFrames and uses Spark implementations.
    """
    
    def __init__(self) -> None:
        """Initialize temporal feature generator."""
        self.logger = get_audit_logger()
    
    def delta(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        periods: int = 1,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute delta (difference from previous period).
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute delta on
            output_col: Output column name (default: {column}_delta)
            time_col: Time column for sorting
            group_cols: Columns to group by
            periods: Number of periods to look back
            
        Returns:
            DataFrame with delta column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.delta(df, column, output_col, time_col, group_cols, periods)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_delta"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        # Compute delta
        if group_cols:
            df[output_col] = df.groupby(group_cols)[column].diff(periods)
        else:
            df[output_col] = df[column].diff(periods)
        
        return df
    
    def pct_change(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        periods: int = 1,
        fill_method: Optional[str] = None,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute percent change from previous period.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute percent change on
            output_col: Output column name (default: {column}_pct_change)
            time_col: Time column for sorting
            group_cols: Columns to group by
            periods: Number of periods to look back
            fill_method: Method to fill NaN values
            
        Returns:
            DataFrame with percent change column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.pct_change(df, column, output_col, time_col, group_cols, periods, fill_method)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_pct_change"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        # Compute percent change
        if group_cols:
            df[output_col] = df.groupby(group_cols)[column].pct_change(periods, fill_method=fill_method)
        else:
            df[output_col] = df[column].pct_change(periods, fill_method=fill_method)
        
        return df
    
    def momentum(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 3,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute momentum (current value - rolling mean).
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute momentum on
            output_col: Output column name (default: {column}_momentum)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            DataFrame with momentum column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.momentum(df, column, output_col, time_col, group_cols, window)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_momentum_{window}"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        # Compute rolling mean
        if group_cols:
            rolling_mean = df.groupby(group_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        else:
            rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
        
        df[output_col] = df[column] - rolling_mean
        
        return df
    
    def volatility(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
        method: str = "std",
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute volatility (rolling standard deviation or coefficient of variation).
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute volatility on
            output_col: Output column name (default: {column}_volatility)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            method: "std" for standard deviation or "cv" for coefficient of variation
            
        Returns:
            DataFrame with volatility column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.volatility(df, column, output_col, time_col, group_cols, window, method)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_volatility_{window}"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        # Compute volatility
        if group_cols:
            if method == "std":
                df[output_col] = df.groupby(group_cols)[column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            elif method == "cv":
                rolling_mean = df.groupby(group_cols)[column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_std = df.groupby(group_cols)[column].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[output_col] = rolling_std / (rolling_mean + 1e-8)
        else:
            if method == "std":
                df[output_col] = df[column].rolling(window=window, min_periods=1).std()
            elif method == "cv":
                rolling_mean = df[column].rolling(window=window, min_periods=1).mean()
                rolling_std = df[column].rolling(window=window, min_periods=1).std()
                df[output_col] = rolling_std / (rolling_mean + 1e-8)
        
        return df
    
    def trend_slope(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute trend slope using linear regression over rolling window.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute trend slope on
            output_col: Output column name (default: {column}_trend_slope)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            DataFrame with trend slope column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.trend_slope(df, column, output_col, time_col, group_cols, window)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_trend_slope_{window}"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        def compute_slope(series):
            """Compute slope via linear regression."""
            if len(series) < 2:
                return np.nan
            
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            
            x = x[mask]
            y = y[mask]
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Compute rolling slope
        if group_cols:
            df[output_col] = df.groupby(group_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(compute_slope, raw=False)
            )
        else:
            df[output_col] = df[column].rolling(window=window, min_periods=2).apply(compute_slope, raw=False)
        
        return df
    
    def rolling_rank(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
        pct: bool = True,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute rolling rank of current value within window.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute rolling rank on
            output_col: Output column name (default: {column}_rolling_rank)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            pct: Whether to return percentile rank (0-1) or absolute rank
            
        Returns:
            DataFrame with rolling rank column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.rolling_rank(df, column, output_col, time_col, group_cols, window)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_rolling_rank_{window}"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        def compute_rank(series):
            """Compute rank of last element in series."""
            if len(series) < 2:
                return np.nan
            
            last_val = series.iloc[-1]
            rank = (series < last_val).sum() + 1
            
            if pct:
                return rank / len(series)
            return rank
        
        # Compute rolling rank
        if group_cols:
            df[output_col] = df.groupby(group_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=1).apply(compute_rank, raw=False)
            )
        else:
            df[output_col] = df[column].rolling(window=window, min_periods=1).apply(compute_rank, raw=False)
        
        return df
    
    def minmax_range(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        time_col: Optional[str] = None,
        group_cols: Optional[List[str]] = None,
        window: int = 6,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Compute range (max - min) over rolling window.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to compute range on
            output_col: Output column name (default: {column}_range)
            time_col: Time column for sorting
            group_cols: Columns to group by
            window: Rolling window size
            
        Returns:
            DataFrame with range column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.minmax_range(df, column, output_col, time_col, group_cols, window)
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_minmax_range_{window}"
        
        df = df.copy()
        
        # Sort by time if provided
        if time_col:
            df = df.sort_values(time_col)
        
        # Compute range
        if group_cols:
            rolling_max = df.groupby(group_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            rolling_min = df.groupby(group_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
        else:
            rolling_max = df[column].rolling(window=window, min_periods=1).max()
            rolling_min = df[column].rolling(window=window, min_periods=1).min()
        
        df[output_col] = rolling_max - rolling_min
        
        return df


class CategoricalEncoder:
    """
    Categorical encoding utilities.
    
    Provides frequency encoding, target mean encoding, and rare category grouping.
    
    Automatically detects Spark DataFrames and uses Spark implementations.
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
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        output_col: Optional[str] = None,
        sample_weight_col: Optional[str] = None,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Encode categorical variable with frequency.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to encode
            output_col: Output column name (default: {column}_freq)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            DataFrame with encoded column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.categorical_encoder import SparkCategoricalEncoder
            spark_encoder = SparkCategoricalEncoder(self.handle_missing)
            result = spark_encoder.freq_encoding(df, column, output_col, sample_weight_col)
            self.mappings.update(spark_encoder.mappings)
            return result
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_freq"
        
        df = df.copy()
        freq_map = df[column].value_counts(normalize=True).to_dict()
        
        df[output_col] = df[column].map(freq_map)
        
        # Handle unseen categories
        df[output_col] = df[output_col].fillna(0)
        
        self.mappings[output_col] = {"type": "frequency", "mapping": freq_map}
        
        return df
    
    def target_mean_encoding(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        target: str,
        output_col: Optional[str] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        sample_weight_col: Optional[str] = None,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Target mean encoding with smoothing.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to encode
            target: Target column
            output_col: Output column name (default: {column}_target_mean)
            smoothing: Smoothing parameter
            min_samples_leaf: Minimum samples per category
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            DataFrame with encoded column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.categorical_encoder import SparkCategoricalEncoder
            spark_encoder = SparkCategoricalEncoder(self.handle_missing)
            result = spark_encoder.target_mean_encoding(
                df, column, target, output_col, smoothing, min_samples_leaf, sample_weight_col
            )
            self.mappings.update(spark_encoder.mappings)
            return result
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_target_mean"
        
        df = df.copy()
        
        # Global mean
        global_mean = df[target].mean()
        
        # Category statistics
        cat_stats = df.groupby(column)[target].agg(['mean', 'count'])
        
        # Smoothed mean
        smoothed_mean = (
            (cat_stats['count'] * cat_stats['mean'] + smoothing * global_mean) /
            (cat_stats['count'] + smoothing)
        )
        
        # Filter by min samples
        smoothed_mean = smoothed_mean[cat_stats['count'] >= min_samples_leaf]
        
        # Map to DataFrame
        df[output_col] = df[column].map(smoothed_mean).fillna(global_mean)
        
        self.mappings[output_col] = {
            "type": "target_mean",
            "mapping": smoothed_mean.to_dict(),
            "global_mean": global_mean,
            "smoothing": smoothing,
        }
        
        return df
    
    def rare_grouping(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        column: str,
        threshold: float = 0.01,
        rare_label: str = "RARE",
        output_col: Optional[str] = None,
        sample_weight_col: Optional[str] = None,
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """
        Group rare categories together.
        
        Args:
            df: Input DataFrame (pandas or Spark)
            column: Column to process
            threshold: Frequency threshold for rare categories
            rare_label: Label for rare categories
            output_col: Output column name (default: {column}_grouped)
            sample_weight_col: Optional sample weight column for compressed data
            
        Returns:
            DataFrame with grouped column
        """
        # Auto-detect Spark DataFrame
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.categorical_encoder import SparkCategoricalEncoder
            spark_encoder = SparkCategoricalEncoder(self.handle_missing)
            result = spark_encoder.rare_grouping(
                df, column, threshold, rare_label, output_col, sample_weight_col
            )
            self.mappings.update(spark_encoder.mappings)
            return result
        
        # Pandas implementation
        if output_col is None:
            output_col = f"{column}_grouped"
        
        df = df.copy()
        
        freq = df[column].value_counts(normalize=True)
        rare_cats = freq[freq < threshold].index
        
        df[output_col] = df[column].apply(lambda x: rare_label if x in rare_cats else x)
        
        self.mappings[output_col] = {
            "type": "rare_grouping",
            "rare_categories": rare_cats.tolist(),
            "threshold": threshold,
        }
        
        return df
    
    def export_mappings(self, path: str) -> None:
        """Export encodings to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump(self.mappings, f, indent=2, default=str)
        self.logger.info(f"Exported encoding mappings to {path}")
