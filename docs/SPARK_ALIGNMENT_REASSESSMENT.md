# Spark Alignment Reassessment

## Executive Summary

**Critical Finding:** The current codebase is **NOT aligned** with the requirement that "most feature engineering should be done in Spark environment given the large data requirement for scorecard."

**Key Misalignments:**
1. ❌ Factory function defaults to `pandas` instead of `spark`
2. ❌ No automatic engine detection based on DataFrame type
3. ❌ Pipeline (`ScorecardPipeline`) is pandas-only
4. ❌ Enhanced features (`TemporalTrendFeatures`, `CategoricalEncoder`, `FeatureValidator`) are pandas-only
5. ❌ No unified interface that auto-detects and routes to appropriate engine

---

## Current State Analysis

### **1. Factory Function Default**

**Location:** `src/cr_score/features/engineering.py:1116`

```python
def create_feature_engineer(
    config: Optional[FeatureEngineeringConfig] = None,
    engine: str = "pandas",  # ❌ WRONG DEFAULT
) -> BaseFeatureEngineer:
```

**Issue:** Defaults to pandas, requiring explicit `engine="spark"` for large-scale operations.

**Impact:** Users must remember to specify Spark, leading to OOM errors on large datasets.

---

### **2. No Automatic Engine Detection**

**Current:** Users must manually specify engine type.

**Missing:** Function that auto-detects DataFrame type and selects appropriate engine:

```python
# What we need:
engineer = create_feature_engineer_auto(df, config)  # Auto-detects Spark vs pandas
```

**Impact:** Error-prone, requires manual decision-making.

---

### **3. Pipeline is Pandas-Only**

**Location:** `src/cr_score/pipeline.py`

**Current State:**
- `ScorecardPipeline` only accepts `pd.DataFrame`
- No Spark DataFrame support
- No automatic conversion to Spark for large datasets

**Code:**
```python
def fit(self, df: pd.DataFrame, target_col: str, ...) -> None:
    # Only works with pandas
```

**Impact:** Cannot use pipeline for large-scale scorecard development.

---

### **4. Enhanced Features are Pandas-Only**

**Location:** `src/cr_score/features/enhanced_features.py`

**Current State:**
- `TemporalTrendFeatures` - pandas only
- `CategoricalEncoder` - pandas only  
- `FeatureValidator` - pandas only
- `DependencyGraph` - works with both (no DataFrame operations)

**Impact:** Cannot use advanced features on Spark DataFrames.

---

### **5. No Unified Interface**

**Current:** Separate classes (`PandasFeatureEngineer`, `SparkFeatureEngineer`)

**Missing:** Unified interface that:
- Auto-detects DataFrame type
- Routes to appropriate implementation
- Provides consistent API regardless of engine

---

## Required Architectural Changes

### **Priority 1: Make Spark the Default**

#### 1.1 Update Factory Function

**Change:**
```python
def create_feature_engineer(
    config: Optional[FeatureEngineeringConfig] = None,
    engine: str = "spark",  # ✅ Change default to spark
) -> BaseFeatureEngineer:
```

**Rationale:** For large-scale scorecard development, Spark should be the default.

**Backward Compatibility:** Users can still specify `engine="pandas"` explicitly.

---

#### 1.2 Add Auto-Detection Function

**New Function:**
```python
def create_feature_engineer_auto(
    df: Union[pd.DataFrame, "SparkDataFrame"],
    config: Optional[FeatureEngineeringConfig] = None,
    prefer_spark: bool = True,
) -> BaseFeatureEngineer:
    """
    Create feature engineer with automatic engine detection.
    
    Args:
        df: Input DataFrame (pandas or Spark)
        config: Feature engineering configuration
        prefer_spark: If True, prefer Spark even for pandas DataFrames (convert if needed)
        
    Returns:
        Feature engineer instance (Spark or Pandas)
        
    Examples:
        >>> # Auto-detect from DataFrame type
        >>> engineer = create_feature_engineer_auto(spark_df, config)
        >>> # Prefer Spark (convert pandas to Spark if needed)
        >>> engineer = create_feature_engineer_auto(pandas_df, config, prefer_spark=True)
    """
    # Detect DataFrame type
    if isinstance(df, SparkDataFrame):
        return SparkFeatureEngineer(config)
    
    # For pandas DataFrames
    if prefer_spark and PYSPARK_AVAILABLE:
        # Convert to Spark for large-scale processing
        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        spark_df = spark.createDataFrame(df)
        return SparkFeatureEngineer(config)
    else:
        return PandasFeatureEngineer(config)
```

---

### **Priority 2: Unified Feature Engineer Interface**

#### 2.1 Create Unified Class

**New Class:**
```python
class FeatureEngineer:
    """
    Unified feature engineer that auto-detects engine.
    
    Automatically uses Spark for large datasets and pandas for small ones.
    Provides consistent API regardless of underlying engine.
    
    Examples:
        >>> # Works with both pandas and Spark
        >>> engineer = FeatureEngineer(config)
        >>> df_transformed = engineer.fit_transform(df)  # Auto-detects
    """
    
    def __init__(
        self,
        config: Optional[FeatureEngineeringConfig] = None,
        engine: Optional[str] = None,  # None = auto-detect
        prefer_spark: bool = True,
    ):
        self.config = config
        self.engine = engine
        self.prefer_spark = prefer_spark
        self._engineer: Optional[BaseFeatureEngineer] = None
    
    def fit_transform(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"]
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Fit and transform with automatic engine detection."""
        if self._engineer is None:
            self._engineer = self._get_engineer(df)
        return self._engineer.fit_transform(df)
    
    def _get_engineer(self, df: Union[pd.DataFrame, "SparkDataFrame"]) -> BaseFeatureEngineer:
        """Get appropriate engineer based on DataFrame type."""
        if self.engine:
            return create_feature_engineer(self.config, engine=self.engine)
        
        # Auto-detect
        if isinstance(df, SparkDataFrame):
            return SparkFeatureEngineer(self.config)
        
        if self.prefer_spark and PYSPARK_AVAILABLE:
            # Convert pandas to Spark for large-scale processing
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            spark_df = spark.createDataFrame(df)
            return SparkFeatureEngineer(self.config)
        
        return PandasFeatureEngineer(self.config)
```

---

### **Priority 3: Pipeline Spark Support**

#### 3.1 Update Pipeline to Support Spark

**Changes to `ScorecardPipeline`:**

```python
class ScorecardPipeline:
    def __init__(
        self,
        ...,
        engine: Optional[str] = None,  # None = auto-detect
        prefer_spark: bool = True,  # Prefer Spark for large datasets
    ):
        self.engine = engine
        self.prefer_spark = prefer_spark
        # ... rest of init
    
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
        elif self.prefer_spark and PYSPARK_AVAILABLE:
            # Convert to Spark for large datasets
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            df = spark.createDataFrame(df)
            self.engine_ = "spark"
        else:
            self.engine_ = "pandas"
        
        # Use appropriate feature engineer
        if self.engine_ == "spark":
            self.feature_engineer_ = SparkFeatureEngineer(...)
        else:
            self.feature_engineer_ = PandasFeatureEngineer(...)
        
        # ... rest of fit logic
```

---

### **Priority 4: Enhanced Features Spark Support**

#### 4.1 Create Spark Versions

**New Files:**
- `src/cr_score/spark/features/temporal_trends.py` - Spark temporal features
- `src/cr_score/spark/features/categorical_encoder.py` - Spark categorical encoding
- `src/cr_score/spark/features/feature_validator.py` - Spark validation

#### 4.2 Unified Interfaces

**Pattern:**
```python
# In enhanced_features.py
class TemporalTrendFeatures:
    """Unified temporal trend features (auto-detects engine)."""
    
    def delta(
        self,
        df: Union[pd.DataFrame, "SparkDataFrame"],
        value_col: str,
        ...
    ) -> Union[pd.DataFrame, "SparkDataFrame"]:
        """Delta calculation with auto-detection."""
        if isinstance(df, SparkDataFrame):
            from cr_score.spark.features.temporal_trends import SparkTemporalTrendFeatures
            spark_trend = SparkTemporalTrendFeatures()
            return spark_trend.delta(df, value_col, ...)
        else:
            # Use existing pandas implementation
            return self._delta_pandas(df, value_col, ...)
```

---

## Updated Enhancement Plan

### **Phase 1: Core Architecture Changes** (CRITICAL)

1. **Change factory default to Spark**
   - Update `create_feature_engineer()` default
   - Add deprecation warning for explicit `engine="pandas"` (optional)

2. **Add auto-detection**
   - Implement `create_feature_engineer_auto()`
   - Implement `FeatureEngineer` unified class

3. **Update pipeline**
   - Add Spark DataFrame support
   - Add auto-detection logic
   - Add `prefer_spark` parameter

**Timeline:** 1-2 days

---

### **Phase 2: Enhanced Features Spark Support** (HIGH PRIORITY)

1. **Create Spark implementations:**
   - `SparkTemporalTrendFeatures`
   - `SparkCategoricalEncoder`
   - `SparkFeatureValidator`

2. **Add unified interfaces:**
   - Update `TemporalTrendFeatures` to auto-detect
   - Update `CategoricalEncoder` to auto-detect
   - Update `FeatureValidator` to auto-detect

**Timeline:** 3-4 days

---

### **Phase 3: Optimization & Metrics** (MEDIUM PRIORITY)

1. **Optimization utilities** (as per original plan)
2. **Metrics collection** (as per original plan)

**Timeline:** 2-3 days

---

## Migration Strategy

### **Backward Compatibility**

1. **Keep existing classes:**
   - `PandasFeatureEngineer` - still available
   - `SparkFeatureEngineer` - still available
   - `create_feature_engineer()` - still works, but defaults to Spark

2. **Add deprecation warnings (optional):**
   ```python
   if engine == "pandas":
       warnings.warn(
           "Default engine changed to 'spark' for large-scale processing. "
           "Specify engine='pandas' explicitly if needed.",
           DeprecationWarning
       )
   ```

3. **Documentation updates:**
   - Update examples to show Spark-first approach
   - Add migration guide for existing code

---

## Testing Strategy

### **Unit Tests:**
- Test auto-detection logic
- Test unified interface with both DataFrame types
- Test pipeline with Spark DataFrames
- Verify backward compatibility

### **Integration Tests:**
- Test end-to-end pipeline with Spark
- Compare results between pandas and Spark
- Test conversion from pandas to Spark

### **Performance Tests:**
- Benchmark Spark vs pandas on large datasets
- Verify Spark provides 10x+ speedup for large data

---

## Success Criteria

1. ✅ Factory function defaults to Spark
2. ✅ Auto-detection works correctly
3. ✅ Pipeline supports Spark DataFrames
4. ✅ Enhanced features work with Spark
5. ✅ Unified interface provides consistent API
6. ✅ Backward compatibility maintained
7. ✅ Documentation updated
8. ✅ All tests passing

---

## Implementation Order

1. **Week 1: Core Architecture**
   - Change factory default
   - Add auto-detection
   - Update pipeline

2. **Week 2: Enhanced Features**
   - Implement Spark versions
   - Add unified interfaces
   - Integration testing

3. **Week 3: Optimization & Polish**
   - Optimization utilities
   - Metrics collection
   - Documentation

---

## Risk Assessment

### **Low Risk:**
- Changing factory default (backward compatible with explicit parameter)
- Adding auto-detection (new functionality, doesn't break existing code)

### **Medium Risk:**
- Pipeline changes (requires thorough testing)
- Enhanced features Spark support (complex implementation)

### **Mitigation:**
- Comprehensive test coverage
- Gradual rollout with feature flags
- Clear migration documentation

---

## Conclusion

The current codebase is **not aligned** with the requirement for Spark-first feature engineering. The proposed changes will:

1. ✅ Make Spark the default for large-scale processing
2. ✅ Provide automatic engine detection
3. ✅ Enable pipeline to work with Spark
4. ✅ Support enhanced features on Spark
5. ✅ Maintain backward compatibility

**Recommendation:** Implement Phase 1 (Core Architecture) immediately, as it's foundational for all other enhancements.

---

**Status:** Reassessment Complete - Ready for Implementation
