"""
Feature engineering toolkit for credit risk modeling.

Provides configurable tools for creating derived features including:
- Time-based aggregations (worst/max/min last N periods)
- Statistical aggregations (mean, std, count, sum)
- Ratio and derived features
- Rolling window features
- Custom transformations

Supports both single feature and batch mode operations.
Works with both pandas DataFrames and PySpark DataFrames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import Window, functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    SparkDataFrame = None


class AggregationType(str, Enum):
    """Types of aggregations available."""
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    SUM = "sum"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    RANGE = "range"  # max - min
    WORST = "worst"  # max for negative metrics like delinquency


class TimeWindow(str, Enum):
    """Common time windows for aggregations."""
    LAST_1M = "last_1_month"
    LAST_3M = "last_3_months"
    LAST_6M = "last_6_months"
    LAST_12M = "last_12_months"
    LAST_24M = "last_24_months"
    ALL_TIME = "all_time"


class MissingStrategy(str, Enum):
    """Strategies for handling missing values."""
    KEEP = "keep"  # Keep NaN as-is
    ZERO = "zero"  # Fill with 0
    MEAN = "mean"  # Fill with mean
    MEDIAN = "median"  # Fill with median
    CONSTANT = "constant"  # Fill with specific value
    FLAG = "flag"  # Create indicator column


class DivideByZeroPolicy(str, Enum):
    """Policies for division by zero."""
    NAN = "nan"  # Return NaN
    ZERO = "zero"  # Return 0
    CONSTANT = "constant"  # Return specific constant


@dataclass
class FeatureRecipe:
    """
    Configuration for a single feature engineering operation.
    
    Args:
        name: Output feature name
        source_cols: Input column(s) to use
        operation: Aggregation or transformation type
        window: Time window for aggregation (optional)
        params: Additional parameters for the operation
        description: Human-readable description
        missing_strategy: Strategy for handling missing values
        missing_value: Value to use if strategy is CONSTANT
        create_missing_indicator: Whether to create missing indicator column
        impute_scope: Scope for imputation (global or group)
        divide_by_zero_policy: Policy for division by zero in ratios
        divide_by_zero_constant: Constant to use if policy is CONSTANT
        depends_on: List of feature names this recipe depends on
        
    Examples:
        >>> # Max delinquency in last 3 months
        >>> recipe = FeatureRecipe(
        ...     name="max_dpd_3m",
        ...     source_cols=["days_past_due"],
        ...     operation=AggregationType.MAX,
        ...     window=TimeWindow.LAST_3M,
        ...     description="Maximum delinquency in last 3 months"
        ... )
        
        >>> # Utilization ratio with missing handling
        >>> recipe = FeatureRecipe(
        ...     name="utilization_ratio",
        ...     source_cols=["balance", "credit_limit"],
        ...     operation="ratio",
        ...     description="Balance / Credit Limit",
        ...     missing_strategy=MissingStrategy.ZERO,
        ...     divide_by_zero_policy=DivideByZeroPolicy.NAN
        ... )
    """
    name: str
    source_cols: Union[str, List[str]]
    operation: Union[AggregationType, str]
    window: Optional[Union[TimeWindow, str]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Missing value handling
    missing_strategy: Union[MissingStrategy, str] = MissingStrategy.KEEP
    missing_value: Optional[Any] = None
    create_missing_indicator: bool = False
    impute_scope: str = "global"  # "global" or "group"
    
    # Divide by zero handling for ratios
    divide_by_zero_policy: Union[DivideByZeroPolicy, str] = DivideByZeroPolicy.NAN
    divide_by_zero_constant: Optional[float] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Normalize source_cols to list."""
        if isinstance(self.source_cols, str):
            self.source_cols = [self.source_cols]
        
        # Normalize enums
        if isinstance(self.missing_strategy, str):
            try:
                self.missing_strategy = MissingStrategy(self.missing_strategy)
            except ValueError:
                pass
        
        if isinstance(self.divide_by_zero_policy, str):
            try:
                self.divide_by_zero_policy = DivideByZeroPolicy(self.divide_by_zero_policy)
            except ValueError:
                pass


@dataclass
class FeatureEngineeringConfig:
    """
    Batch configuration for multiple feature engineering operations.
    
    Args:
        recipes: List of feature recipes to apply
        id_col: Column identifying unique entities (e.g., customer_id)
        time_col: Column with timestamps (optional, for time-based features)
        group_cols: Columns to group by for aggregations (optional)
        enable_caching: Whether to cache intermediate results (Spark only)
        validate_dependencies: Whether to validate dependency graph
        
    Examples:
        >>> config = FeatureEngineeringConfig(
        ...     recipes=[
        ...         FeatureRecipe("max_dpd_3m", "dpd", AggregationType.MAX, TimeWindow.LAST_3M),
        ...         FeatureRecipe("avg_balance_6m", "balance", AggregationType.MEAN, TimeWindow.LAST_6M),
        ...         FeatureRecipe("util_ratio", ["balance", "limit"], "ratio"),
        ...     ],
        ...     id_col="customer_id",
        ...     time_col="snapshot_date"
        ... )
    """
    recipes: List[FeatureRecipe]
    id_col: Optional[str] = None
    time_col: Optional[str] = None
    group_cols: Optional[List[str]] = None
    enable_caching: bool = False
    validate_dependencies: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FeatureEngineeringConfig":
        """Create config from dictionary."""
        recipes = [
            FeatureRecipe(**recipe) if isinstance(recipe, dict) else recipe
            for recipe in config.get("recipes", [])
        ]
        return cls(
            recipes=recipes,
            id_col=config.get("id_col"),
            time_col=config.get("time_col"),
            group_cols=config.get("group_cols"),
            enable_caching=config.get("enable_caching", False),
            validate_dependencies=config.get("validate_dependencies", True),
        )


@dataclass
class FeatureMetadata:
    """Metadata for a single feature."""
    name: str
    source_columns: List[str]
    operation: str
    parameters: Dict[str, Any]
    window: Optional[str]
    missing_strategy: str
    dependencies: List[str]
    engine: str
    created_timestamp: str
    output_dtype: str
    execution_time_ms: Optional[float] = None


class FeatureRegistry:
    """
    Registry for tracking feature metadata and lineage.
    
    Maintains a record of all features created, their dependencies,
    and lineage for audit and reproducibility purposes.
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self.features: Dict[str, FeatureMetadata] = {}
        self.logger = get_audit_logger()
    
    def register(
        self,
        name: str,
        source_columns: List[str],
        operation: str,
        parameters: Dict[str, Any],
        window: Optional[str],
        missing_strategy: str,
        dependencies: List[str],
        engine: str,
        output_dtype: str,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """Register a feature."""
        metadata = FeatureMetadata(
            name=name,
            source_columns=source_columns,
            operation=operation,
            parameters=parameters,
            window=window,
            missing_strategy=missing_strategy,
            dependencies=dependencies,
            engine=engine,
            created_timestamp=datetime.utcnow().isoformat(),
            output_dtype=output_dtype,
            execution_time_ms=execution_time_ms,
        )
        
        self.features[name] = metadata
        self.logger.info(f"Registered feature: {name}", feature_metadata=metadata.__dict__)
    
    def get(self, name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature."""
        return self.features.get(name)
    
    def export_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export registry as dictionary."""
        return {
            name: {
                k: v for k, v in metadata.__dict__.items()
            }
            for name, metadata in self.features.items()
        }
    
    def export_json(self, path: str) -> None:
        """Export registry to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.export_dict(), f, indent=2)
        
        self.logger.info(f"Exported feature registry to {path}")
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """Get dependencies for a feature."""
        metadata = self.get(feature_name)
        if metadata:
            return metadata.dependencies
        return []
    
    def get_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get full lineage for a feature."""
        metadata = self.get(feature_name)
        if not metadata:
            return {}
        
        lineage = {
            "feature": feature_name,
            "sources": metadata.source_columns,
            "operation": metadata.operation,
            "dependencies": metadata.dependencies,
            "dependency_tree": {},
        }
        
        # Recursively get dependencies
        for dep in metadata.dependencies:
            lineage["dependency_tree"][dep] = self.get_lineage(dep)
        
        return lineage


class BaseFeatureEngineer(ABC):
    """Base class for feature engineering."""
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = get_audit_logger()
        self.created_features_: List[str] = []
        
    @abstractmethod
    def transform(self, df: Union[pd.DataFrame, "SparkDataFrame"]) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Apply feature engineering transformations."""
        pass
    
    @abstractmethod
    def fit_transform(self, df: Union[pd.DataFrame, "SparkDataFrame"]) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Fit and transform in one step."""
        pass


class PandasFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for pandas DataFrames.
    
    Supports single feature or batch mode via config.
    
    Examples:
        >>> # Batch mode with config
        >>> config = FeatureEngineeringConfig(
        ...     recipes=[
        ...         FeatureRecipe("max_dpd_3m", "dpd", AggregationType.MAX),
        ...         FeatureRecipe("avg_util", "utilization", AggregationType.MEAN),
        ...     ],
        ...     id_col="customer_id"
        ... )
        >>> engineer = PandasFeatureEngineer(config)
        >>> df_transformed = engineer.fit_transform(df)
        
        >>> # Single feature mode
        >>> engineer = PandasFeatureEngineer()
        >>> df = engineer.create_aggregation(
        ...     df, "max_dpd_3m", "dpd", AggregationType.MAX,
        ...     group_by="customer_id", window_months=3
        ... )
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply configured feature engineering.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        if self.config is None:
            raise ValueError("Config required for batch transform. Use single feature methods instead.")
        
        df_result = df.copy()
        
        for recipe in self.config.recipes:
            self.logger.info(f"Creating feature: {recipe.name}", operation=recipe.operation)
            
            try:
                df_result = self._apply_recipe(df_result, recipe)
                self.created_features_.append(recipe.name)
            except Exception as e:
                self.logger.error(f"Failed to create feature {recipe.name}: {e}")
                raise
        
        self.logger.info(f"Created {len(self.created_features_)} features", features=self.created_features_)
        return df_result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform (stateless for now)."""
        return self.transform(df)
    
    def _apply_recipe(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Apply a single feature recipe."""
        operation = recipe.operation
        
        # Normalize operation string to enum if applicable
        normalized_op = self._normalize_operation(operation)
        
        # Standard aggregations
        if normalized_op in AggregationType.__members__.values():
            return self._apply_aggregation(df, recipe)
        
        # Custom operations (keep as strings)
        if operation == "ratio":
            return self._create_ratio(df, recipe)
        elif operation == "difference":
            return self._create_difference(df, recipe)
        elif operation == "product":
            return self._create_product(df, recipe)
        elif operation == "log":
            return self._create_log(df, recipe)
        elif operation == "sqrt":
            return self._create_sqrt(df, recipe)
        elif operation == "clip":
            return self._create_clip(df, recipe)
        elif operation == "bin":
            return self._create_bin(df, recipe)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _normalize_operation(self, operation: Union[AggregationType, str]) -> AggregationType:
        """Normalize operation string to enum."""
        if isinstance(operation, AggregationType):
            return operation
        
        # Try to convert string to enum
        if isinstance(operation, str):
            try:
                return AggregationType(operation)
            except ValueError:
                # Not a standard aggregation, return as-is
                return operation
        
        return operation
    
    def _apply_aggregation(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Apply aggregation operation."""
        source_col = recipe.source_cols[0]
        
        # Normalize operation to enum
        operation = self._normalize_operation(recipe.operation)
        
        if not self.config or not self.config.group_cols:
            # Simple aggregation without grouping
            df = df.copy()
            df[recipe.name] = self._compute_aggregation(df[source_col], operation)
            return df
        
        # Group-based aggregation
        group_cols = self.config.group_cols
        
        # Filter by time window if specified
        if recipe.window and self.config.time_col:
            df_filtered = self._filter_time_window(df, recipe.window)
        else:
            df_filtered = df
        
        # Compute aggregation
        agg_result = df_filtered.groupby(group_cols)[source_col].agg(
            self._get_agg_func(operation)
        ).reset_index()
        agg_result.columns = list(group_cols) + [recipe.name]
        
        # Merge back
        df = df.merge(agg_result, on=group_cols, how="left")
        
        return df
    
    def _filter_time_window(self, df: pd.DataFrame, window: Union[TimeWindow, str]) -> pd.DataFrame:
        """Filter DataFrame by time window."""
        if not self.config or not self.config.time_col:
            return df
        
        time_col = self.config.time_col
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Get window in months
        window_months = self._get_window_months(window)
        
        if window_months is None:
            return df
        
        max_date = df[time_col].max()
        cutoff_date = max_date - pd.DateOffset(months=window_months)
        
        return df[df[time_col] >= cutoff_date]
    
    def _get_window_months(self, window: Union[TimeWindow, str]) -> Optional[int]:
        """Convert window to months."""
        if window == TimeWindow.LAST_1M:
            return 1
        elif window == TimeWindow.LAST_3M:
            return 3
        elif window == TimeWindow.LAST_6M:
            return 6
        elif window == TimeWindow.LAST_12M:
            return 12
        elif window == TimeWindow.LAST_24M:
            return 24
        elif window == TimeWindow.ALL_TIME:
            return None
        
        # Try to parse from string
        if isinstance(window, str):
            if "month" in window.lower():
                try:
                    return int(''.join(filter(str.isdigit, window)))
                except ValueError:
                    pass
        
        return None
    
    def _compute_aggregation(self, series: pd.Series, operation: Union[AggregationType, str]) -> pd.Series:
        """Compute aggregation on series."""
        # Normalize to enum
        if isinstance(operation, str):
            try:
                operation = AggregationType(operation)
            except ValueError:
                raise ValueError(f"Unknown aggregation: {operation}")
        
        if operation == AggregationType.MAX or operation == AggregationType.WORST:
            return series.max()
        elif operation == AggregationType.MIN:
            return series.min()
        elif operation == AggregationType.MEAN:
            return series.mean()
        elif operation == AggregationType.MEDIAN:
            return series.median()
        elif operation == AggregationType.STD:
            return series.std()
        elif operation == AggregationType.SUM:
            return series.sum()
        elif operation == AggregationType.COUNT:
            return series.count()
        elif operation == AggregationType.FIRST:
            return series.iloc[0] if len(series) > 0 else np.nan
        elif operation == AggregationType.LAST:
            return series.iloc[-1] if len(series) > 0 else np.nan
        elif operation == AggregationType.RANGE:
            return series.max() - series.min()
        else:
            raise ValueError(f"Unknown aggregation: {operation}")
    
    def _get_agg_func(self, operation: Union[AggregationType, str]) -> Union[str, Callable]:
        """Get pandas aggregation function name."""
        # Convert string to enum if needed
        if isinstance(operation, str):
            try:
                operation = AggregationType(operation)
            except ValueError:
                # If not a valid enum, return the string as-is
                return operation
        
        if operation == AggregationType.WORST:
            return "max"
        elif operation == AggregationType.RANGE:
            return lambda x: x.max() - x.min()
        else:
            return operation.value
    
    def _create_ratio(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create ratio feature."""
        if len(recipe.source_cols) != 2:
            raise ValueError("Ratio requires exactly 2 source columns")
        
        df = df.copy()
        numerator, denominator = recipe.source_cols
        
        # Handle division by zero
        df[recipe.name] = np.where(
            df[denominator] != 0,
            df[numerator] / df[denominator],
            np.nan
        )
        
        return df
    
    def _create_difference(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create difference feature."""
        if len(recipe.source_cols) != 2:
            raise ValueError("Difference requires exactly 2 source columns")
        
        df = df.copy()
        col1, col2 = recipe.source_cols
        df[recipe.name] = df[col1] - df[col2]
        
        return df
    
    def _create_product(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create product feature."""
        df = df.copy()
        df[recipe.name] = df[recipe.source_cols].prod(axis=1)
        
        return df
    
    def _create_log(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create log-transformed feature."""
        df = df.copy()
        source_col = recipe.source_cols[0]
        
        # Add 1 to handle zeros if specified
        add_one = recipe.params.get("add_one", True)
        
        if add_one:
            df[recipe.name] = np.log1p(df[source_col])
        else:
            df[recipe.name] = np.log(df[source_col])
        
        return df
    
    def _create_sqrt(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create square root feature."""
        df = df.copy()
        source_col = recipe.source_cols[0]
        df[recipe.name] = np.sqrt(np.maximum(df[source_col], 0))
        
        return df
    
    def _create_clip(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create clipped feature."""
        df = df.copy()
        source_col = recipe.source_cols[0]
        
        lower = recipe.params.get("lower", None)
        upper = recipe.params.get("upper", None)
        
        df[recipe.name] = df[source_col].clip(lower=lower, upper=upper)
        
        return df
    
    def _create_bin(self, df: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Create binned feature."""
        df = df.copy()
        source_col = recipe.source_cols[0]
        
        bins = recipe.params.get("bins", 10)
        labels = recipe.params.get("labels", None)
        
        df[recipe.name] = pd.cut(df[source_col], bins=bins, labels=labels)
        
        return df
    
    # Convenience methods for single feature creation
    
    def create_aggregation(
        self,
        df: pd.DataFrame,
        feature_name: str,
        source_col: str,
        operation: AggregationType,
        group_by: Optional[Union[str, List[str]]] = None,
        window_months: Optional[int] = None,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a single aggregation feature.
        
        Args:
            df: Input DataFrame
            feature_name: Name for new feature
            source_col: Column to aggregate
            operation: Aggregation type
            group_by: Column(s) to group by
            window_months: Time window in months (requires time_col)
            time_col: Column with timestamps
            
        Returns:
            DataFrame with new feature
            
        Examples:
            >>> # Max delinquency last 3 months per customer
            >>> df = engineer.create_aggregation(
            ...     df, "max_dpd_3m", "days_past_due", AggregationType.MAX,
            ...     group_by="customer_id", window_months=3, time_col="date"
            ... )
            
            >>> # Average balance per customer
            >>> df = engineer.create_aggregation(
            ...     df, "avg_balance", "balance", AggregationType.MEAN,
            ...     group_by="customer_id"
            ... )
        """
        window = None
        if window_months:
            window = f"last_{window_months}_months"
        
        recipe = FeatureRecipe(
            name=feature_name,
            source_cols=source_col,
            operation=operation,
            window=window,
        )
        
        if group_by:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
        else:
            group_cols = None
        
        temp_config = FeatureEngineeringConfig(
            recipes=[recipe],
            group_cols=group_cols,
            time_col=time_col,
        )
        
        temp_engineer = PandasFeatureEngineer(temp_config)
        return temp_engineer.transform(df)
    
    def create_ratio(
        self,
        df: pd.DataFrame,
        feature_name: str,
        numerator_col: str,
        denominator_col: str,
    ) -> pd.DataFrame:
        """
        Create a ratio feature.
        
        Args:
            df: Input DataFrame
            feature_name: Name for new feature
            numerator_col: Numerator column
            denominator_col: Denominator column
            
        Returns:
            DataFrame with new feature
            
        Examples:
            >>> # Utilization ratio
            >>> df = engineer.create_ratio(
            ...     df, "utilization", "balance", "credit_limit"
            ... )
        """
        recipe = FeatureRecipe(
            name=feature_name,
            source_cols=[numerator_col, denominator_col],
            operation="ratio",
        )
        
        return self._create_ratio(df, recipe)
    
    def create_rolling_feature(
        self,
        df: pd.DataFrame,
        feature_name: str,
        source_col: str,
        window: int,
        operation: AggregationType,
        group_by: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Create rolling window feature.
        
        Args:
            df: Input DataFrame
            feature_name: Name for new feature
            source_col: Column to aggregate
            window: Rolling window size
            operation: Aggregation type
            group_by: Column(s) to group by
            
        Returns:
            DataFrame with new feature
            
        Examples:
            >>> # Rolling 3-period average
            >>> df = engineer.create_rolling_feature(
            ...     df, "balance_ma3", "balance", 3, AggregationType.MEAN,
            ...     group_by="customer_id"
            ... )
        """
        df = df.copy()
        
        agg_func = self._get_agg_func(operation)
        
        if group_by:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
            df[feature_name] = df.groupby(group_cols)[source_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).agg(agg_func)
            )
        else:
            df[feature_name] = df[source_col].rolling(window=window, min_periods=1).agg(agg_func)
        
        return df


class SparkFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for PySpark DataFrames.
    
    Optimized for distributed processing of large datasets.
    
    Examples:
        >>> # Batch mode with config
        >>> config = FeatureEngineeringConfig(
        ...     recipes=[
        ...         FeatureRecipe("max_dpd_3m", "dpd", AggregationType.MAX),
        ...         FeatureRecipe("avg_util", "utilization", AggregationType.MEAN),
        ...     ],
        ...     id_col="customer_id"
        ... )
        >>> engineer = SparkFeatureEngineer(config)
        >>> df_transformed = engineer.fit_transform(df)
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        """Initialize Spark feature engineer."""
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark not available. Install pyspark to use SparkFeatureEngineer.")
        
        super().__init__(config)
    
    def transform(self, df: "SparkDataFrame") -> "SparkDataFrame":
        """
        Apply configured feature engineering.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            DataFrame with new features
        """
        if self.config is None:
            raise ValueError("Config required for batch transform. Use single feature methods instead.")
        
        for recipe in self.config.recipes:
            self.logger.info(f"Creating feature: {recipe.name}", operation=recipe.operation)
            
            try:
                df = self._apply_recipe(df, recipe)
                self.created_features_.append(recipe.name)
            except Exception as e:
                self.logger.error(f"Failed to create feature {recipe.name}: {e}")
                raise
        
        self.logger.info(f"Created {len(self.created_features_)} features", features=self.created_features_)
        return df
    
    def fit_transform(self, df: "SparkDataFrame") -> "SparkDataFrame":
        """Fit and transform (stateless for now)."""
        return self.transform(df)
    
    def _apply_recipe(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Apply a single feature recipe."""
        operation = recipe.operation
        
        # Normalize operation string to enum if applicable
        normalized_op = self._normalize_operation(operation)
        
        # Standard aggregations
        if normalized_op in AggregationType.__members__.values():
            return self._apply_aggregation(df, recipe)
        
        # Custom operations
        if operation == "ratio":
            return self._create_ratio(df, recipe)
        elif operation == "difference":
            return self._create_difference(df, recipe)
        elif operation == "product":
            return self._create_product(df, recipe)
        elif operation == "log":
            return self._create_log(df, recipe)
        elif operation == "sqrt":
            return self._create_sqrt(df, recipe)
        elif operation == "clip":
            return self._create_clip(df, recipe)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _normalize_operation(self, operation: Union[AggregationType, str]) -> Union[AggregationType, str]:
        """Normalize operation string to enum."""
        if isinstance(operation, AggregationType):
            return operation
        
        # Try to convert string to enum
        if isinstance(operation, str):
            try:
                return AggregationType(operation)
            except ValueError:
                # Not a standard aggregation, return as-is
                return operation
        
        return operation
    
    def _apply_aggregation(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Apply aggregation operation."""
        source_col = recipe.source_cols[0]
        
        # Normalize operation to enum
        operation = self._normalize_operation(recipe.operation)
        
        if not self.config or not self.config.group_cols:
            # Simple aggregation without grouping
            agg_expr = self._get_agg_expr(operation, source_col)
            return df.withColumn(recipe.name, agg_expr)
        
        # Group-based aggregation
        group_cols = self.config.group_cols
        
        # Filter by time window if specified
        if recipe.window and self.config.time_col:
            df_filtered = self._filter_time_window(df, recipe.window)
        else:
            df_filtered = df
        
        # Compute aggregation
        agg_expr = self._get_agg_expr(operation, source_col)
        agg_result = df_filtered.groupBy(*group_cols).agg(
            agg_expr.alias(recipe.name)
        )
        
        # Join back
        df = df.join(agg_result, on=group_cols, how="left")
        
        return df
    
    def _filter_time_window(self, df: "SparkDataFrame", window: Union[TimeWindow, str]) -> "SparkDataFrame":
        """Filter DataFrame by time window."""
        if not self.config or not self.config.time_col:
            return df
        
        time_col = self.config.time_col
        window_months = self._get_window_months(window)
        
        if window_months is None:
            return df
        
        max_date = df.agg(F.max(time_col)).collect()[0][0]
        cutoff_date = F.add_months(F.lit(max_date), -window_months)
        
        return df.filter(F.col(time_col) >= cutoff_date)
    
    def _get_window_months(self, window: Union[TimeWindow, str]) -> Optional[int]:
        """Convert window to months."""
        if window == TimeWindow.LAST_1M:
            return 1
        elif window == TimeWindow.LAST_3M:
            return 3
        elif window == TimeWindow.LAST_6M:
            return 6
        elif window == TimeWindow.LAST_12M:
            return 12
        elif window == TimeWindow.LAST_24M:
            return 24
        elif window == TimeWindow.ALL_TIME:
            return None
        
        # Try to parse from string
        if isinstance(window, str):
            if "month" in window.lower():
                try:
                    return int(''.join(filter(str.isdigit, window)))
                except ValueError:
                    pass
        
        return None
    
    def _get_agg_expr(self, operation: Union[AggregationType, str], col_name: str):
        """Get Spark aggregation expression."""
        # Normalize to enum
        if isinstance(operation, str):
            try:
                operation = AggregationType(operation)
            except ValueError:
                raise ValueError(f"Unknown aggregation: {operation}")
        
        col = F.col(col_name)
        
        if operation == AggregationType.MAX or operation == AggregationType.WORST:
            return F.max(col)
        elif operation == AggregationType.MIN:
            return F.min(col)
        elif operation == AggregationType.MEAN:
            return F.mean(col)
        elif operation == AggregationType.MEDIAN:
            return F.expr(f"percentile_approx({col_name}, 0.5)")
        elif operation == AggregationType.STD:
            return F.stddev(col)
        elif operation == AggregationType.SUM:
            return F.sum(col)
        elif operation == AggregationType.COUNT:
            return F.count(col)
        elif operation == AggregationType.FIRST:
            return F.first(col)
        elif operation == AggregationType.LAST:
            return F.last(col)
        elif operation == AggregationType.RANGE:
            return F.max(col) - F.min(col)
        else:
            raise ValueError(f"Unknown aggregation: {operation}")
    
    def _create_ratio(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create ratio feature."""
        if len(recipe.source_cols) != 2:
            raise ValueError("Ratio requires exactly 2 source columns")
        
        numerator, denominator = recipe.source_cols
        
        return df.withColumn(
            recipe.name,
            F.when(F.col(denominator) != 0, F.col(numerator) / F.col(denominator))
            .otherwise(None)
        )
    
    def _create_difference(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create difference feature."""
        if len(recipe.source_cols) != 2:
            raise ValueError("Difference requires exactly 2 source columns")
        
        col1, col2 = recipe.source_cols
        return df.withColumn(recipe.name, F.col(col1) - F.col(col2))
    
    def _create_product(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create product feature."""
        from functools import reduce
        
        product = reduce(lambda a, b: a * b, [F.col(c) for c in recipe.source_cols])
        return df.withColumn(recipe.name, product)
    
    def _create_log(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create log-transformed feature."""
        source_col = recipe.source_cols[0]
        add_one = recipe.params.get("add_one", True)
        
        if add_one:
            return df.withColumn(recipe.name, F.log1p(F.col(source_col)))
        else:
            return df.withColumn(recipe.name, F.log(F.col(source_col)))
    
    def _create_sqrt(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create square root feature."""
        source_col = recipe.source_cols[0]
        return df.withColumn(
            recipe.name,
            F.sqrt(F.when(F.col(source_col) >= 0, F.col(source_col)).otherwise(0))
        )
    
    def _create_clip(self, df: "SparkDataFrame", recipe: FeatureRecipe) -> "SparkDataFrame":
        """Create clipped feature."""
        source_col = recipe.source_cols[0]
        lower = recipe.params.get("lower", None)
        upper = recipe.params.get("upper", None)
        
        col_expr = F.col(source_col)
        
        if lower is not None:
            col_expr = F.when(col_expr < lower, lower).otherwise(col_expr)
        if upper is not None:
            col_expr = F.when(col_expr > upper, upper).otherwise(col_expr)
        
        return df.withColumn(recipe.name, col_expr)
    
    def create_aggregation(
        self,
        df: "SparkDataFrame",
        feature_name: str,
        source_col: str,
        operation: AggregationType,
        group_by: Optional[Union[str, List[str]]] = None,
        window_months: Optional[int] = None,
        time_col: Optional[str] = None,
    ) -> "SparkDataFrame":
        """
        Create a single aggregation feature.
        
        Args:
            df: Input Spark DataFrame
            feature_name: Name for new feature
            source_col: Column to aggregate
            operation: Aggregation type
            group_by: Column(s) to group by
            window_months: Time window in months (requires time_col)
            time_col: Column with timestamps
            
        Returns:
            DataFrame with new feature
            
        Examples:
            >>> # Max delinquency last 3 months per customer
            >>> df = engineer.create_aggregation(
            ...     df, "max_dpd_3m", "days_past_due", AggregationType.MAX,
            ...     group_by="customer_id", window_months=3, time_col="date"
            ... )
        """
        window = None
        if window_months:
            window = f"last_{window_months}_months"
        
        recipe = FeatureRecipe(
            name=feature_name,
            source_cols=source_col,
            operation=operation,
            window=window,
        )
        
        if group_by:
            group_cols = [group_by] if isinstance(group_by, str) else group_by
        else:
            group_cols = None
        
        temp_config = FeatureEngineeringConfig(
            recipes=[recipe],
            group_cols=group_cols,
            time_col=time_col,
        )
        
        temp_engineer = SparkFeatureEngineer(temp_config)
        return temp_engineer.transform(df)


def create_feature_engineer(
    config: Optional[FeatureEngineeringConfig] = None,
    engine: str = "spark",
) -> BaseFeatureEngineer:
    """
    Factory function to create feature engineer.
    
    Args:
        config: Feature engineering configuration
        engine: "pandas" or "spark" (default: "spark" for large-scale processing)
        
    Returns:
        Feature engineer instance
        
    Note:
        Default engine changed to "spark" for large-scale scorecard development.
        Specify engine="pandas" explicitly if needed for small datasets.
        
    Examples:
        >>> config = FeatureEngineeringConfig(...)
        >>> # Default: Spark engine
        >>> engineer = create_feature_engineer(config)
        >>> df_transformed = engineer.fit_transform(df)
        
        >>> # Explicit pandas engine
        >>> engineer = create_feature_engineer(config, engine="pandas")
        >>> df_transformed = engineer.fit_transform(df)
    """
    if engine == "pandas":
        return PandasFeatureEngineer(config)
    elif engine == "spark":
        if not PYSPARK_AVAILABLE:
            raise ImportError(
                "PySpark not available. Install pyspark to use Spark engine, "
                "or specify engine='pandas' for pandas-based processing."
            )
        return SparkFeatureEngineer(config)
    else:
        raise ValueError(f"Unknown engine: {engine}. Choose 'pandas' or 'spark'.")


def create_feature_engineer_auto(
    df: Union[pd.DataFrame, "SparkDataFrame"],
    config: Optional[FeatureEngineeringConfig] = None,
    prefer_spark: bool = True,
) -> BaseFeatureEngineer:
    """
    Create feature engineer with automatic engine detection.
    
    Automatically detects DataFrame type and selects appropriate engine.
    For pandas DataFrames, can optionally convert to Spark for large-scale processing.
    
    Args:
        df: Input DataFrame (pandas or Spark)
        config: Feature engineering configuration
        prefer_spark: If True, prefer Spark even for pandas DataFrames (convert if needed)
        
    Returns:
        Feature engineer instance (Spark or Pandas)
        
    Examples:
        >>> # Auto-detect from DataFrame type
        >>> engineer = create_feature_engineer_auto(spark_df, config)
        >>> df_transformed = engineer.fit_transform(spark_df)
        
        >>> # Prefer Spark (convert pandas to Spark if needed)
        >>> engineer = create_feature_engineer_auto(pandas_df, config, prefer_spark=True)
        >>> df_transformed = engineer.fit_transform(pandas_df)
    """
    # Detect DataFrame type
    if isinstance(df, SparkDataFrame):
        if not PYSPARK_AVAILABLE:
            raise ImportError("Spark DataFrame provided but PySpark not available.")
        return SparkFeatureEngineer(config)
    
    # For pandas DataFrames
    if prefer_spark and PYSPARK_AVAILABLE:
        # Return Spark engineer (conversion happens in FeatureEngineer class)
        return SparkFeatureEngineer(config)
    else:
        return PandasFeatureEngineer(config)


class FeatureEngineer:
    """
    Unified feature engineer that auto-detects engine.
    
    Automatically uses Spark for large datasets and pandas for small ones.
    Provides consistent API regardless of underlying engine.
    
    Examples:
        >>> # Works with both pandas and Spark
        >>> engineer = FeatureEngineer(config)
        >>> df_transformed = engineer.fit_transform(df)  # Auto-detects
        
        >>> # Force specific engine
        >>> engineer = FeatureEngineer(config, engine="spark")
        >>> df_transformed = engineer.fit_transform(df)
    """
    
    def __init__(
        self,
        config: Optional[FeatureEngineeringConfig] = None,
        engine: Optional[str] = None,  # None = auto-detect
        prefer_spark: bool = True,
    ):
        """
        Initialize unified feature engineer.
        
        Args:
            config: Feature engineering configuration
            engine: Explicit engine ("pandas" or "spark"), None for auto-detection
            prefer_spark: If True, prefer Spark even for pandas DataFrames
        """
        self.config = config
        self.engine = engine
        self.prefer_spark = prefer_spark
        self._engineer: Optional[BaseFeatureEngineer] = None
        self._detected_engine: Optional[str] = None
        self.logger = get_audit_logger()
    
    def fit_transform(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"]
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Fit and transform with automatic engine detection."""
        if self._engineer is None:
            self._engineer = self._get_engineer(df)
        
        # Convert pandas to Spark if needed
        if isinstance(df, pd.DataFrame) and self._detected_engine == "spark":
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder.appName("CR_Score_FeatureEngineering").getOrCreate()
            df = spark.createDataFrame(df)
        
        return self._engineer.fit_transform(df)
    
    def transform(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"]
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Transform with automatic engine detection."""
        if self._engineer is None:
            self._engineer = self._get_engineer(df)
        
        # Convert pandas to Spark if needed
        if isinstance(df, pd.DataFrame) and self._detected_engine == "spark":
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder.appName("CR_Score_FeatureEngineering").getOrCreate()
            df = spark.createDataFrame(df)
        
        return self._engineer.transform(df)
    
    def _get_engineer(self, df: Union[pd.DataFrame, "SparkDataFrame"]) -> BaseFeatureEngineer:
        """Get appropriate engineer based on DataFrame type."""
        if self.engine:
            self._detected_engine = self.engine
            return create_feature_engineer(self.config, engine=self.engine)
        
        # Auto-detect
        if isinstance(df, SparkDataFrame):
            self._detected_engine = "spark"
            self.logger.info("Auto-detected Spark engine from DataFrame type")
            return SparkFeatureEngineer(self.config)
        
        if self.prefer_spark and PYSPARK_AVAILABLE:
            self._detected_engine = "spark"
            self.logger.info("Auto-selected Spark engine (prefer_spark=True)")
            return SparkFeatureEngineer(self.config)
        
        self._detected_engine = "pandas"
        self.logger.info("Auto-selected Pandas engine")
        return PandasFeatureEngineer(self.config)
    
    @property
    def detected_engine(self) -> Optional[str]:
        """Get detected engine (None if not yet detected)."""
        return self._detected_engine
