================================================================================
CR_Score CODING STANDARDS
================================================================================

Version: 1.0
Date: 2026-01-15
Language: Python 3.9+
Purpose: Detailed coding conventions and patterns for CR_Score implementation

================================================================================
1. PYTHON STYLE & PEP 8 EXTENSIONS
================================================================================

CORE PEP 8 COMPLIANCE:
   - Line length: 100 characters (not 79)
   - Indentation: 4 spaces (no tabs)
   - Imports: Sorted alphabetically within groups
   - Naming: snake_case for functions/vars, PascalCase for classes
   - Docstrings: Triple quotes, never single-line comments for modules

CR_Score EXTENSIONS:

Line Breaks:
   - Break lines at 100 chars for readability
   - Binary operators: break before operator (PEP 8 recommended style)
   - Use parentheses for implicit line continuation, not backslash
   
    # Good:
    result = (
        df.filter(col("age") > 18)
        .select("name", "score")
        .where(col("score") > 600)
    )
    
    # Bad:
    result = df.filter(col("age") > 18) \
        .select("name", "score")

Magic Numbers:
   - All non-obvious numbers are named constants at module top
   - Exception: Loop indices (i, j), probability 0.5, 1.0 scaling factors
   - Example:
   
    DEFAULT_SHUFFLE_PARTITIONS = 800
    MIN_BIN_EVENTS = 30
    MAX_BIN_PURITY_TOLERANCE = 0.05
    
    # In code:
    df = df.repartition(DEFAULT_SHUFFLE_PARTITIONS)

Comments:
   - Only explain WHY, never WHAT
   - Don't comment obvious code
   - Use # for inline, """ """ for block
   - Update comments when code changes
   
    # Good:
    # Skip records with missing target to avoid undefined direction
    df_clean = df.filter(col("target").isNotNull())
    
    # Bad:
    # Filter out nulls
    df_clean = df.filter(col("target").isNotNull())

================================================================================
2. FUNCTION SIGNATURE STANDARDS
================================================================================

Type Hints REQUIRED:

    from typing import Optional, Dict, List, Tuple, Union
    from pathlib import Path
    
    def process_binning(
        df: "pyspark.sql.DataFrame",
        config: "BinningConfig",
        run_id: str,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, "BinObject"]:
        """
        Process binning for all variables in config.
        
        Args:
            df: Input Spark DataFrame with target and features
            config: Binning configuration
            run_id: Unique run identifier for logging and artifacts
            output_dir: Optional output directory (defaults to run artifact dir)
            
        Returns:
            Dictionary mapping variable names to BinObject instances
            
        Raises:
            BinningError: If configuration invalid or process fails
            FileNotFoundError: If output_dir does not exist
            
        Example:
            >>> bins = process_binning(df, config, "run_20260115_001")
            >>> bins["age"].summary()
        """

Union Types:
   - Use Union for multiple acceptable types
   - Order by priority: most specific first
   - Use | operator (Python 3.10+) or Union for older versions
   
    def parse_value(val: Union[int, float, str]) -> float:
        """Parse numeric or string value to float."""

Optional Shorthand:
   - Use Optional[T] instead of Union[T, None]
   - Always provide default=None if Optional parameter

Generic Types:
   - Use List[T], Dict[K, V], Tuple[T, ...] from typing
   - Be specific: not List but List[str]
   - Never use list, dict, tuple lowercase (Python 3.9+ deprecation)

Custom Types:
   - Create TypeAlias for complex repeated types
   
    from typing import TypeAlias
    EventRate: TypeAlias = float  # Range [0, 1]
    Weight: TypeAlias = float     # Range [0, ∞)

================================================================================
3. DOCSTRING SPECIFICATION
================================================================================

Format: Google-style docstrings (standardized across team)

Template:

    def complex_function(
        param1: str,
        param2: Optional[int] = None,
    ) -> Dict[str, float]:
        """One-line summary of function purpose.
        
        Extended description explaining:
        - What the function does
        - When to use it
        - Any important side effects or assumptions
        
        Config Parameters:
            Optional config settings that affect behavior:
            - max_bins (int): Maximum bins per variable, default 10
            - method (str): Binning method, default "equal_width"
        
        Args:
            param1: Description of param1. Explain type constraints.
            param2: Optional description. Explain when to use.
            
        Returns:
            Description of return value. Explain structure and meaning.
            Example: {"bin_1": 0.523, "bin_2": 0.477} representing event rates.
            
        Raises:
            ConfigurationError: If param1 and param2 conflict
            ValueError: If param2 < 0
            
        Example:
            >>> result = complex_function("test", param2=5)
            >>> print(result)
            {'bin_1': 0.523, 'bin_2': 0.477}
            
        Note:
            This function uses Spark internally for large datasets (>1M rows).
            For smaller datasets, use simple_function() instead.
            
        See Also:
            related_function: For alternative approach
        """

Sections ALWAYS included:
   - One-line summary
   - Args
   - Returns
   - Raises (only if function raises custom exceptions)
   - Example (for public APIs)

Optional but recommended:
   - Config Parameters
   - Note (important caveats)
   - See Also (related functions)

================================================================================
4. CONFIGURATION HANDLING PATTERNS
================================================================================

Pydantic Model for Config:

    from pydantic import BaseModel, Field, validator
    from typing import List, Optional
    
    class BinningConfig(BaseModel):
        """Configuration for binning step."""
        
        method: str = Field(
            default="equal_width",
            description="Binning method: equal_width, quantile, or custom"
        )
        max_bins: int = Field(
            default=10,
            ge=2,  # Greater than or equal to 2
            le=100,
            description="Maximum number of bins per variable"
        )
        enforce_monotonic: bool = Field(
            default=True,
            description="Enforce event rate monotonicity across bins"
        )
        min_events_per_bin: int = Field(
            default=30,
            description="Minimum events required in each bin"
        )
        
        @validator("method")
        def validate_method(cls, v):
            if v not in ["equal_width", "quantile", "custom"]:
                raise ValueError(f"Unknown method: {v}")
            return v
        
        class Config:
            """Pydantic configuration."""
            frozen = False  # Allow updates after instantiation
            extra = "forbid"  # Reject unknown fields
            
    # Usage:
    config = BinningConfig(**yaml.safe_load(open("config.yml")))

Config Loading Pattern:

    from CR_Score.core.config import load_and_validate_config
    
    def run_step(config_path: str, run_id: str) -> None:
        """Execute step with config validation."""
        config = load_and_validate_config(config_path, schema=BinningConfig)
        logger = get_audit_logger(run_id)
        logger.info(f"Loaded config: {config.dict()}")
        # ... rest of logic

================================================================================
5. SPARK DATAFRAME OPERATIONS
================================================================================

Best Practices:

VECTORIZED OPERATIONS:
   - Never use .apply() or .map() on large DataFrames
   - Use pyspark.sql.functions instead
   
    # Good:
    from pyspark.sql.functions import col, when, round
    df_processed = (
        df
        .withColumn("age_bin", when(col("age") < 30, "young")
                              .when(col("age") < 60, "mid")
                              .otherwise("senior"))
        .withColumn("score_round", round(col("score"), 2))
    )
    
    # Bad:
    def assign_age_bin(age):
        if age < 30: return "young"
        elif age < 60: return "mid"
        else: return "senior"
    df_processed = df.withColumn("age_bin", 
        col("age").apply(assign_age_bin))

AGGREGATION WITH WEIGHTS:
   - All aggregations respect sample_weight parameter
   - Use named aggregation functions
   
    from pyspark.sql.functions import sum as F_sum, count, mean, variance
    
    df_agg = (
        df
        .groupBy("bin_id", "segment")
        .agg(
            F_sum("sample_weight").alias("total_weight"),
            F_sum(col("event") * col("sample_weight")).alias("total_events"),
            (F_sum(col("event") * col("sample_weight")) / 
             F_sum("sample_weight")).alias("event_rate")
        )
    )

PARTITIONING & SKEW HANDLING:
   - Set partition count via config
   - Detect skew via partition size histogram
   - Use salting for highly skewed join keys
   
    def apply_salting(df: DataFrame, key_col: str, salt_factor: int = 8):
        """Add salt to skewed column before join/group by."""
        from pyspark.sql.functions import rand, concat, lit
        
        df_salted = (
            df
            .withColumn("_salt", (rand() * salt_factor).cast("int"))
            .withColumn(key_col, concat(col(key_col), lit("_"), col("_salt")))
        )
        return df_salted

CHECKPOINTING:
   - Checkpoint long DAGs (>10 stages)
   - Checkpoint before expensive operations
   
    if df.rdd.getNumPartitions() > 800:
        df = df.checkpoint()  # Breaks DAG, saves to checkpoint_dir

CACHING:
   - Cache only intermediate results used multiple times
   - Use MEMORY_AND_DISK for spill safety
   - Unpersist after use
   
    from pyspark import StorageLevel
    df_cached = df.persist(StorageLevel.MEMORY_AND_DISK)
    # ... use df_cached multiple times
    df_cached.unpersist()

NULL HANDLING:
   - Explicit null handling, never ambiguous
   
    # Good:
    df_filtered = (
        df
        .filter(col("age").isNotNull())
        .filter(col("target").isin([0, 1]))
    )
    
    # Bad:
    df_filtered = df.filter(col("age") != None)  # Wrong!

================================================================================
6. ERROR HANDLING & CUSTOM EXCEPTIONS
================================================================================

Exception Hierarchy:

    class CR_ScoreException(Exception):
        """Base exception for CR_Score."""
        
    class ConfigurationError(CR_ScoreException):
        """Raised when configuration is invalid."""
        
    class BinningError(CR_ScoreException):
        """Raised when binning fails."""
        
    class DataValidationError(CR_ScoreException):
        """Raised when data fails validation."""
        
    class CompressionError(CR_ScoreException):
        """Raised when compression fails or verification fails."""

Error Handling Pattern:

    from CR_Score.core.logging import get_audit_logger
    
    def risky_operation(config_path: str) -> None:
        """Execute operation with proper error handling."""
        logger = get_audit_logger("run_id")
        
        try:
            config = load_config(config_path)
            validate_config(config)
            result = process(config)
            logger.info("Operation completed successfully")
            return result
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise  # Re-raise after logging
            
        except IOError as e:
            logger.error(f"IO error reading {config_path}: {e}")
            raise CR_ScoreException(f"Failed to read config: {e}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise CR_ScoreException("Unexpected error during processing") from e

Never Catch-All:
   - Avoid bare except:
   - Avoid broad except Exception: unless immediately re-raised
   - Be specific about caught exceptions

================================================================================
7. LOGGING STANDARDS
================================================================================

Logger Import:

    from CR_Score.core.logging import get_audit_logger
    
    logger = get_audit_logger(__name__)  # In module

Logging Levels:

    logger.debug("Variable X = {x}")     # Development debugging, verbose
    logger.info("Step completed in {t}s") # User-facing progress
    logger.warning("Rare value dropped")  # Unexpected but non-critical
    logger.error("Failed to write file")  # Error that stops process
    logger.critical("DB unreachable")     # System-level failure

Structured Logging (JSON):

    logger.info(
        "binning_completed",
        extra={
            "run_id": "run_20260115_001",
            "step_id": "03_binning",
            "variable": "age",
            "num_bins": 12,
            "iv": 0.523,
            "execution_time_s": 45.2,
        }
    )
    
    # Output: {"timestamp": "...", "level": "INFO", "message": "binning_completed", 
    #          "run_id": "...", ...}

Avoid in Logs:
   - Sensitive data (SSN, credit card, etc.)
   - Passwords or secrets
   - PII unless anonymized
   - Highly verbose output (use debug level)

================================================================================
8. ARTIFACT HANDLING PATTERNS
================================================================================

Artifact Output Pattern:

    def run_step(config: Config, run_id: str, output_dir: Path) -> None:
        """Execute step and save artifacts."""
        from CR_Score.core.registry import artifact_index
        from CR_Score.core.hashing import compute_hash
        import json
        from datetime import datetime
        
        logger = get_audit_logger(run_id)
        
        # Execute core logic
        result = process_data(config)
        
        # Save artifacts
        artifact_file = output_dir / f"step_output.json"
        artifact_file.write_text(json.dumps(result, indent=2))
        
        # Compute hash
        content_hash = compute_hash(artifact_file)
        
        # Index artifact
        artifact_meta = {
            "artifact_id": f"{run_id}/03_binning_tables",
            "artifact_type": "binning_table",
            "file_paths": [str(artifact_file)],
            "content_hash": content_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "schema_version": "1.0",
            "lineage": ["run_id/01_eda"],
        }
        artifact_index.register(artifact_meta)
        logger.info("Artifact registered", extra=artifact_meta)

Reading Artifacts:

    def read_and_verify(artifact_id: str, expected_hash: Optional[str] = None):
        """Read artifact and optionally verify hash."""
        from CR_Score.core.registry import artifact_index
        from CR_Score.core.hashing import compute_hash
        
        meta = artifact_index.get(artifact_id)
        if not meta:
            raise FileNotFoundError(f"Artifact {artifact_id} not found")
        
        # Load content
        content = json.loads(Path(meta["file_paths"][0]).read_text())
        
        # Verify hash if provided
        if expected_hash and meta["content_hash"] != expected_hash:
            raise ValueError(f"Hash mismatch for {artifact_id}")
        
        return content

================================================================================
9. TEST PATTERNS
================================================================================

Unit Test Template:

    import pytest
    from unittest.mock import Mock, patch, MagicMock
    from CR_Score.binning import process_binning
    
    class TestProcessBinning:
        """Tests for process_binning function."""
        
        @pytest.fixture
        def sample_df(self, spark):
            """Create sample DataFrame for testing."""
            return spark.createDataFrame([
                (25, 1), (35, 0), (45, 1), (55, 0),
            ], ["age", "target"])
        
        @pytest.fixture
        def binning_config(self):
            """Create test binning config."""
            return BinningConfig(
                method="equal_width",
                max_bins=3,
                min_events_per_bin=1,
            )
        
        def test_basic_binning(self, sample_df, binning_config):
            """Test basic binning with simple input."""
            result = process_binning(sample_df, binning_config, "test_run")
            assert len(result) == 1
            assert "age" in result
        
        def test_empty_dataframe(self, spark, binning_config):
            """Test with empty DataFrame."""
            empty_df = spark.createDataFrame([], "age int, target int")
            with pytest.raises(BinningError):
                process_binning(empty_df, binning_config, "test_run")
        
        def test_missing_target_column(self, spark, binning_config):
            """Test when target column is missing."""
            bad_df = spark.createDataFrame([(25,)], ["age"])
            with pytest.raises(DataValidationError):
                process_binning(bad_df, binning_config, "test_run")
        
        @pytest.mark.parametrize("method", ["equal_width", "quantile"])
        def test_different_methods(self, sample_df, method):
            """Test binning with different methods."""
            config = BinningConfig(method=method)
            result = process_binning(sample_df, config, "test_run")
            assert result is not None

Integration Test Template:

    import pytest
    from CR_Score.eda import run_eda
    from CR_Score.binning import process_binning
    
    def test_eda_to_binning_workflow(sample_df, config, run_id):
        """Test end-to-end workflow from EDA to binning."""
        # Step 1: EDA
        eda_results = run_eda(sample_df, config.eda, run_id)
        assert eda_results is not None
        
        # Step 2: Binning
        bins = process_binning(sample_df, config.binning, run_id)
        assert len(bins) > 0
        
        # Verify lineage
        for var_name, bin_obj in bins.items():
            assert bin_obj.lineage == f"{run_id}/01_eda"

================================================================================
10. DETERMINISM & REPRODUCIBILITY
================================================================================

Random Seed Management:

    import numpy as np
    from pyspark.sql.functions import rand
    
    def set_all_seeds(seed: int):
        """Set seeds for all random sources."""
        np.random.seed(seed)
        random.seed(seed)
        # Note: Spark seed set via spark.sql.shuffle.partitions + deterministic operations

Deterministic Ordering:

    # Always sort before operations to ensure reproducibility
    df_sorted = df.sort("id", "timestamp")
    bins_sorted = sorted(bins.items(), key=lambda x: x[0])

Avoid:
   - Using datetime.now() as seed (use run_id + config hash instead)
   - External service calls without mocking in tests
   - Non-deterministic aggregations (only use sum, count, not median/percentile directly)
   - Floating-point comparisons without tolerance

================================================================================
