# Spark Default Implementation Plan

## Overview

Make Spark the default engine for all feature engineering and compression operations, ensuring large-scale scorecard development works seamlessly.

---

## Implementation Phases

### **Phase 1: Core Factory & Auto-Detection** (Priority: CRITICAL)

#### 1.1 Change Factory Default
**File:** `src/cr_score/features/engineering.py`
- Change `create_feature_engineer(engine="pandas")` → `engine="spark"`
- Add backward compatibility note in docstring
- Update function signature

#### 1.2 Add Auto-Detection Function
**File:** `src/cr_score/features/engineering.py`
- Implement `create_feature_engineer_auto()` function
- Auto-detect DataFrame type (pandas vs Spark)
- Support `prefer_spark=True` parameter
- Handle Spark session creation if needed

#### 1.3 Create Unified FeatureEngineer Class
**File:** `src/cr_score/features/engineering.py`
- Implement `FeatureEngineer` class (unified interface)
- Auto-detection on first `fit_transform()` call
- Consistent API regardless of engine
- Store detected engine for reuse

**Estimated Time:** 2-3 hours

---

### **Phase 2: Pipeline Spark Support** (Priority: CRITICAL)

#### 2.1 Update ScorecardPipeline
**File:** `src/cr_score/pipeline.py`
- Change `fit(df: pd.DataFrame)` → `fit(df: Union[pd.DataFrame, SparkDataFrame])`
- Add `engine: Optional[str] = None` parameter to `__init__`
- Add `prefer_spark: bool = True` parameter to `__init__`
- Implement auto-detection logic in `fit()`
- Convert pandas to Spark if `prefer_spark=True` and PySpark available
- Update `predict()` to support Spark DataFrames
- Update `predict_proba()` to support Spark DataFrames

#### 2.2 Update Internal Components
**File:** `src/cr_score/pipeline.py`
- Ensure `AutoBinner` works with Spark (or convert to pandas for binning)
- Ensure `WoEEncoder` works with Spark (or convert to pandas)
- Ensure model training works (convert to pandas if needed for sklearn models)
- Handle Spark DataFrame conversion at appropriate stages

**Estimated Time:** 4-5 hours

---

### **Phase 3: Enhanced Features Spark Support** (Priority: HIGH)

#### 3.1 Create Spark Features Directory
**Files:**
- `src/cr_score/spark/features/__init__.py` (NEW)
- `src/cr_score/spark/features/temporal_trends.py` (NEW)
- `src/cr_score/spark/features/categorical_encoder.py` (NEW)
- `src/cr_score/spark/features/feature_validator.py` (NEW)

#### 3.2 Implement SparkTemporalTrendFeatures
**File:** `src/cr_score/spark/features/temporal_trends.py`
- `delta()` - Window-based delta with Spark Window functions
- `pct_change()` - Percent change with null handling
- `momentum()` - Last value vs window mean
- `volatility()` - Std/CV over window
- `trend_slope()` - Linear regression slope (Spark ML or UDF)
- `rolling_rank()` - Rank within window
- `minmax_range()` - Max - min over window

#### 3.3 Implement SparkCategoricalEncoder
**File:** `src/cr_score/spark/features/categorical_encoder.py`
- `freq_encoding()` - Frequency encoding with broadcast join
- `target_mean_encoding()` - Smoothed target mean with broadcast
- `rare_grouping()` - Group rare categories

#### 3.4 Implement SparkFeatureValidator
**File:** `src/cr_score/spark/features/feature_validator.py`
- `validate_features()` - Compute metrics using Spark aggregations
- `compute_psi()` - PSI calculation with Spark
- `to_dataframe()` - Convert results to pandas (small output)

#### 3.5 Update Enhanced Features to Auto-Detect
**File:** `src/cr_score/features/enhanced_features.py`
- Update `TemporalTrendFeatures` methods to auto-detect Spark
- Update `CategoricalEncoder` methods to auto-detect Spark
- Update `FeatureValidator` methods to auto-detect Spark

**Estimated Time:** 2-3 days

---

### **Phase 4: Compression Spark Support** (Priority: HIGH)

#### 4.1 Verify PostBinningCompressor
**File:** `src/cr_score/spark/compression/post_binning_exact.py`
- Already supports Spark - verify it's the default
- Ensure it's used by default in pipeline when Spark is active
- Add auto-detection if pandas DataFrame passed

#### 4.2 Integration with Pipeline
**File:** `src/cr_score/pipeline.py`
- Use `PostBinningCompressor` automatically when Spark engine is active
- Convert pandas to Spark before compression if needed
- Ensure compression happens before model training

**Estimated Time:** 1-2 hours

---

### **Phase 5: Testing** (Priority: CRITICAL)

#### 5.1 Unit Tests
**Files:**
- `tests/unit/test_feature_engineering_spark.py` (NEW)
- `tests/unit/test_pipeline_spark.py` (NEW)
- `tests/unit/test_enhanced_features_spark.py` (NEW)

**Coverage:**
- Factory function with Spark default
- Auto-detection logic
- Unified FeatureEngineer class
- Pipeline with Spark DataFrames
- Enhanced features with Spark
- Compression with Spark

#### 5.2 Integration Tests
**File:** `tests/integration/test_spark_default.py` (NEW)
- End-to-end pipeline with Spark
- Compare results pandas vs Spark (small data)
- Performance benchmarks (large data)

#### 5.3 Backward Compatibility Tests
- Verify existing pandas code still works
- Verify explicit `engine="pandas"` works
- Verify no breaking changes

**Estimated Time:** 1-2 days

---

### **Phase 6: Documentation** (Priority: HIGH)

#### 6.1 Jupyter Playbook
**File:** `playbooks/10_spark_default_feature_engineering.ipynb` (NEW)
- Demonstrate Spark as default
- Show auto-detection in action
- Compare pandas vs Spark performance
- Show compression with Spark
- Show enhanced features with Spark
- Show pipeline with Spark

#### 6.2 Sphinx Documentation Updates
**Files:**
- `docs/source/api/features.rst` - Update with Spark default
- `docs/source/api/pipeline.rst` - Update with Spark support
- `docs/source/guides/enhanced_features.rst` - Add Spark examples
- `docs/source/guides/quickstart.rst` - Update examples to show Spark
- `docs/source/index.rst` - Update feature list

**Content:**
- Spark-first approach explanation
- Auto-detection documentation
- Migration guide from pandas to Spark
- Performance considerations
- Best practices

**Estimated Time:** 1 day

---

## Detailed Implementation Steps

### Step 1: Change Factory Default

```python
# src/cr_score/features/engineering.py

def create_feature_engineer(
    config: Optional[FeatureEngineeringConfig] = None,
    engine: str = "spark",  # Changed from "pandas"
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
```

---

### Step 2: Add Auto-Detection Function

```python
# src/cr_score/features/engineering.py

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
        # Convert to Spark for large-scale processing
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = SparkSession.builder.appName("CR_Score_FeatureEngineering").getOrCreate()
        # Note: Conversion happens in FeatureEngineer class, not here
        return SparkFeatureEngineer(config)
    else:
        return PandasFeatureEngineer(config)
```

---

### Step 3: Create Unified FeatureEngineer Class

```python
# src/cr_score/features/engineering.py

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
```

---

### Step 4: Update Pipeline

```python
# src/cr_score/pipeline.py

class ScorecardPipeline:
    def __init__(
        self,
        ...,
        engine: Optional[str] = None,  # None = auto-detect
        prefer_spark: bool = True,  # Prefer Spark for large datasets
    ):
        ...
        self.engine = engine
        self.prefer_spark = prefer_spark
        self.engine_: Optional[str] = None  # Detected engine
    
    def fit(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        target_col: str,
        ...
    ) -> None:
        """Fit pipeline with automatic engine detection."""
        # Detect engine
        if isinstance(df, SparkDataFrame):
            self.engine_ = "spark"
        elif self.engine:
            self.engine_ = self.engine
        elif self.prefer_spark and PYSPARK_AVAILABLE:
            # Convert to Spark for large datasets
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder.appName("CR_Score_Pipeline").getOrCreate()
            df = spark.createDataFrame(df)
            self.engine_ = "spark"
        else:
            self.engine_ = "pandas"
        
        # Store original DataFrame type for predict()
        self._input_is_spark = isinstance(df, SparkDataFrame)
        
        # Convert to pandas for binning/modeling if needed
        if self.engine_ == "spark":
            # Use Spark for feature engineering
            # Convert to pandas for binning (OptBinning requires pandas)
            df_pandas = df.toPandas()
        else:
            df_pandas = df
        
        # ... rest of fit logic with df_pandas
```

---

## Testing Checklist

### Unit Tests
- [ ] Factory function defaults to Spark
- [ ] Auto-detection works for Spark DataFrames
- [ ] Auto-detection works for pandas DataFrames
- [ ] Unified FeatureEngineer auto-detects correctly
- [ ] Pipeline accepts Spark DataFrames
- [ ] Pipeline auto-detects engine
- [ ] Enhanced features work with Spark
- [ ] Compression works with Spark

### Integration Tests
- [ ] End-to-end pipeline with Spark
- [ ] Results match between pandas and Spark (small data)
- [ ] Performance improvement with Spark (large data)
- [ ] Backward compatibility maintained

### Documentation
- [ ] Playbook created and tested
- [ ] Sphinx docs updated
- [ ] Examples updated
- [ ] Migration guide added

---

## Timeline

**Total Estimated Time:** 5-7 days

- **Day 1:** Phase 1 (Factory & Auto-Detection)
- **Day 2:** Phase 2 (Pipeline Support)
- **Day 3-4:** Phase 3 (Enhanced Features Spark)
- **Day 4:** Phase 4 (Compression)
- **Day 5:** Phase 5 (Testing)
- **Day 6:** Phase 6 (Documentation)

---

## Success Criteria

1. ✅ Factory function defaults to Spark
2. ✅ Auto-detection works correctly
3. ✅ Pipeline supports Spark DataFrames
4. ✅ Enhanced features work with Spark
5. ✅ Compression uses Spark by default
6. ✅ All tests passing
7. ✅ Playbook created
8. ✅ Documentation updated
9. ✅ Backward compatibility maintained

---

**Status:** Plan Complete - Ready for Implementation
