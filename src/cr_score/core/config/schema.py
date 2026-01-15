"""
Configuration schema validation using Pydantic.

Implements the configuration structure from URD Appendix B with full validation,
defaults, and type checking.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from cr_score.core.exceptions import ConfigValidationError


class ExecutionEngine(str, Enum):
    """Supported execution engines."""

    PYTHON_LOCAL = "python_local"
    SPARK_LOCAL = "spark_local"
    SPARK_CLUSTER = "spark_cluster"


class SplitMethod(str, Enum):
    """Data split strategies."""

    TIME_BASED = "time_based"
    RANDOM = "random"
    STRATIFIED = "stratified"


class CompressionMode(str, Enum):
    """Data compression modes."""

    POST_BINNING_EXACT = "post_binning_exact"
    EDA_SUFFICIENT_STATS = "eda_sufficient_stats"
    HYBRID_TOPK_TAIL = "hybrid_topk_tail"


class BinningMethod(str, Enum):
    """Binning algorithms."""

    QUANTILE = "quantile"
    EQUAL_WIDTH = "equal_width"
    DECISION_TREE = "decision_tree"
    CUSTOM = "custom"


class RejectInferenceMethod(str, Enum):
    """Reject inference strategies."""

    NONE = "none"
    PARCELING = "parceling"
    REWEIGHTING = "reweighting"
    AUGMENTATION = "augmentation"


class ModelType(str, Enum):
    """Supported model types."""

    LOGISTIC = "logistic"
    REGULARIZED_LOGISTIC = "regularized_logistic"


class SparkConfig(BaseModel):
    """Spark execution configuration."""

    shuffle_partitions: int = Field(default=800, ge=1)
    persist_level: str = Field(default="MEMORY_AND_DISK")
    checkpoint_enabled: bool = Field(default=True)
    checkpoint_dir: Optional[str] = None
    skew_mitigation_enabled: bool = Field(default=True, alias="skew_mitigation.enabled")
    skew_mitigation_salting_factor: int = Field(default=8, ge=1, alias="skew_mitigation.salting_factor")

    class Config:
        populate_by_name = True


class ExecutionConfig(BaseModel):
    """Execution engine configuration."""

    engine: ExecutionEngine = Field(default=ExecutionEngine.PYTHON_LOCAL)
    spark: Optional[SparkConfig] = Field(default_factory=SparkConfig)


class DataSource(BaseModel):
    """Data source specification."""

    path: str
    format: str = Field(default="parquet")
    options: Dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    """Data source and schema configuration."""

    sources: List[DataSource]
    schema_contract: Optional[str] = None
    snapshot_ref: Optional[str] = None
    primary_key: Optional[str] = None
    time_key: Optional[str] = None
    segment_keys: List[str] = Field(default_factory=list)


class TargetConfig(BaseModel):
    """Target variable definition."""

    definition: str
    horizon_months: Optional[int] = Field(default=None, ge=1)
    cure_logic: Optional[str] = None


class SplitConfig(BaseModel):
    """Train/test split configuration."""

    method: SplitMethod = Field(default=SplitMethod.TIME_BASED)
    dev_period: Optional[str] = None
    oot_period: Optional[str] = None
    val_period: Optional[str] = None
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    random_seed: int = Field(default=42)


class HighCardinalityConfig(BaseModel):
    """High cardinality handling."""

    enabled: bool = Field(default=True)
    rare_level_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    max_levels: int = Field(default=500, ge=10)


class DataOptimizationConfig(BaseModel):
    """Data optimization settings."""

    column_pruning: bool = Field(default=True)
    type_optimization: bool = Field(default=True)
    missing_normalization: bool = Field(default=True)
    high_cardinality: HighCardinalityConfig = Field(default_factory=HighCardinalityConfig)


class CompressionTrigger(BaseModel):
    """Compression trigger conditions."""

    min_rows: int = Field(default=5000000, ge=0)


class CompressionVerification(BaseModel):
    """Compression verification settings."""

    enabled: bool = Field(default=True)
    tolerance: float = Field(default=0.0, ge=0.0)


class CompressionConfig(BaseModel):
    """Data compression configuration."""

    enabled: bool = Field(default=True)
    mode: CompressionMode = Field(default=CompressionMode.POST_BINNING_EXACT)
    trigger: CompressionTrigger = Field(default_factory=CompressionTrigger)
    verification: CompressionVerification = Field(default_factory=CompressionVerification)


class SamplingConfig(BaseModel):
    """Data sampling configuration."""

    enabled: bool = Field(default=False)
    method: str = Field(default="stratified")
    strata: List[str] = Field(default_factory=list)
    max_rows: Optional[int] = Field(default=None, ge=1000)


class EDAConfig(BaseModel):
    """Exploratory Data Analysis configuration."""

    segments: List[str] = Field(default_factory=list)
    drift_metrics: List[str] = Field(default_factory=lambda: ["PSI", "CSI"])


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    recipes: List[Dict[str, Any]] = Field(default_factory=list)
    selection_filters: Dict[str, Any] = Field(default_factory=dict)


class FineBinningConfig(BaseModel):
    """Fine binning settings."""

    method: BinningMethod = Field(default=BinningMethod.QUANTILE)
    max_bins: int = Field(default=20, ge=2)


class CoarseBinningConfig(BaseModel):
    """Coarse binning settings."""

    monotonicity: bool = Field(default=True)
    min_bin_pct: float = Field(default=0.05, ge=0.0, le=1.0)


class BinningConfig(BaseModel):
    """Binning configuration."""

    fine: FineBinningConfig = Field(default_factory=FineBinningConfig)
    coarse: CoarseBinningConfig = Field(default_factory=CoarseBinningConfig)
    overrides: Dict[str, Any] = Field(default_factory=dict)


class RejectInferenceConfig(BaseModel):
    """Reject inference configuration."""

    method: RejectInferenceMethod = Field(default=RejectInferenceMethod.NONE)
    params: Dict[str, Any] = Field(default_factory=dict)


class RegularizationConfig(BaseModel):
    """Model regularization settings."""

    type: Optional[str] = Field(default=None)
    alpha: float = Field(default=1.0, ge=0.0)
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Model training configuration."""

    type: ModelType = Field(default=ModelType.LOGISTIC)
    regularization: Optional[RegularizationConfig] = None
    feature_selection: Dict[str, Any] = Field(default_factory=dict)
    random_seed: int = Field(default=42)


class CalibrationConfig(BaseModel):
    """Score calibration configuration."""

    enabled: bool = Field(default=True)
    by_segment: bool = Field(default=False)


class ScalingConfig(BaseModel):
    """Score scaling configuration."""

    pdo: int = Field(default=20, ge=1)
    base_score: int = Field(default=600, ge=0)
    base_odds: float = Field(default=50.0, gt=0.0)


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    formats: List[str] = Field(default_factory=lambda: ["html"])
    confluence_safe: bool = Field(default=True)


class InterfacesConfig(BaseModel):
    """Interface enablement."""

    cli: bool = Field(default=True)
    sdk: bool = Field(default=True)
    api: bool = Field(default=True)
    ui: bool = Field(default=True)


class ToolsConfig(BaseModel):
    """MCP tools configuration."""

    enabled_tools: List[str] = Field(default_factory=list)
    permissions: Dict[str, Any] = Field(default_factory=dict)


class ProjectConfig(BaseModel):
    """Project metadata."""

    name: str
    owner: str
    template: str = Field(default="intermediate")
    description: str = Field(default="")
    tags: List[str] = Field(default_factory=list)


class Config(BaseModel):
    """
    Complete CR_Score configuration schema.

    Validates all configuration parameters according to URD v1.2 Appendix B.
    Provides sensible defaults for optional parameters.

    Example:
        >>> config = Config.parse_file("config.yml")
        >>> assert config.execution.engine == ExecutionEngine.SPARK_CLUSTER
        >>> assert config.compression.enabled
    """

    project: ProjectConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    data: DataConfig
    target: TargetConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    data_optimization: DataOptimizationConfig = Field(default_factory=DataOptimizationConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    binning: BinningConfig = Field(default_factory=BinningConfig)
    reject_inference: RejectInferenceConfig = Field(default_factory=RejectInferenceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @field_validator("data")
    @classmethod
    def validate_data_sources(cls, v: DataConfig) -> DataConfig:
        """Ensure at least one data source is provided."""
        if not v.sources:
            raise ValueError("At least one data source must be specified")
        return v

    @model_validator(mode="after")
    def validate_spark_config(self) -> "Config":
        """Validate Spark config when Spark engine is used."""
        if self.execution.engine in [ExecutionEngine.SPARK_LOCAL, ExecutionEngine.SPARK_CLUSTER]:
            if not self.execution.spark:
                raise ValueError(f"Spark config required for engine {self.execution.engine}")
            if self.execution.spark.checkpoint_enabled and not self.execution.spark.checkpoint_dir:
                raise ValueError("checkpoint_dir required when checkpoint_enabled=True")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self.model_dump(mode="python", exclude_none=False)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Export configuration to YAML file.

        Args:
            path: Output file path
        """
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(path: Union[str, Path]) -> Config:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to configuration YAML file

    Returns:
        Validated Config instance

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file not found

    Example:
        >>> config = load_config("config.yml")
        >>> print(config.project.name)
    """
    try:
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        return Config(**raw_config)
    except FileNotFoundError:
        raise ConfigValidationError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ConfigValidationError(f"Config validation failed: {e}", details={"path": str(path)})


def validate_config(config_dict: Dict[str, Any]) -> Config:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        Validated Config instance

    Raises:
        ConfigValidationError: If validation fails

    Example:
        >>> config_dict = {"project": {"name": "test", "owner": "user"}, ...}
        >>> config = validate_config(config_dict)
    """
    try:
        return Config(**config_dict)
    except Exception as e:
        raise ConfigValidationError(f"Config validation failed: {e}")
