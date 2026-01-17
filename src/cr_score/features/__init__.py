"""Feature engineering and selection."""

from cr_score.features.selection import (
    ForwardSelector,
    BackwardSelector,
    StepwiseSelector,
    ExhaustiveSelector,
)

from cr_score.features.engineering import (
    AggregationType,
    TimeWindow,
    MissingStrategy,
    DivideByZeroPolicy,
    FeatureRecipe,
    FeatureEngineeringConfig,
    FeatureMetadata,
    FeatureRegistry,
    PandasFeatureEngineer,
    SparkFeatureEngineer,
    FeatureEngineer,
    create_feature_engineer,
    create_feature_engineer_auto,
)

from cr_score.features.enhanced_features import (
    DependencyGraph,
    FeatureValidator,
    CategoricalEncoder,
    TemporalTrendFeatures,
)

__all__ = [
    # Selection
    "ForwardSelector",
    "BackwardSelector",
    "StepwiseSelector",
    "ExhaustiveSelector",
    # Engineering - Core
    "AggregationType",
    "TimeWindow",
    "MissingStrategy",
    "DivideByZeroPolicy",
    "FeatureRecipe",
    "FeatureEngineeringConfig",
    "FeatureMetadata",
    "FeatureRegistry",
    "PandasFeatureEngineer",
    "SparkFeatureEngineer",
    "FeatureEngineer",
    "create_feature_engineer",
    "create_feature_engineer_auto",
    # Engineering - Enhanced
    "DependencyGraph",
    "FeatureValidator",
    "CategoricalEncoder",
    "TemporalTrendFeatures",
]
