# Spark Enhancement Assessment

## Current State Analysis

### ✅ **What Exists:**

1. **Spark Session Management** (`spark/session.py`)
   - `SparkSessionFactory` - Config-driven session creation
   - Support for local and cluster modes
   - Proper configuration management

2. **Basic Feature Engineering** (`features/engineering.py`)
   - `SparkFeatureEngineer` class exists
   - Supports basic aggregations (MAX, MIN, MEAN, etc.)
   - Supports simple operations (ratio, difference, product, log, sqrt, clip)
   - **Gap**: Missing temporal trend features, categorical encoding

3. **Compression** (`spark/compression/post_binning_exact.py`)
   - `PostBinningCompressor` - Post-binning exact compression
   - Sample weighting support
   - Verification logic

### ❌ **What's Missing (Empty Folders):**

1. **`spark/metrics/`** - EMPTY
   - No Spark-specific metrics collection
   - No performance monitoring for Spark operations
   - No execution statistics tracking

2. **`spark/optimization/`** - EMPTY
   - No caching utilities
   - No partition optimization
   - No skew detection/handling
   - No query optimization helpers

3. **Enhanced Features Spark Support** (`features/enhanced_features.py`)
   - `TemporalTrendFeatures` - **Pandas only** (needs Spark version)
   - `CategoricalEncoder` - **Pandas only** (needs Spark version)
   - `FeatureValidator` - **Pandas only** (needs Spark version)
   - `DependencyGraph` - Works with both (no changes needed)

---

## Required Enhancements

### **Priority 1: Enhanced Features Spark Support**

#### 1.1 Spark Temporal Trend Features

**Location:** `spark/features/temporal_trends.py` (NEW)

**Required Methods:**
- `delta()` - Window-based delta calculation
- `pct_change()` - Percent change with null handling
- `momentum()` - Last value vs window mean
- `volatility()` - Std/CV over window
- `trend_slope()` - Linear regression slope (using Spark ML)
- `rolling_rank()` - Rank within window
- `minmax_range()` - Max - min over window

**Key Requirements:**
- Use Spark Window functions for efficiency
- Support time_col + group_cols partitioning
- Handle nulls consistently with pandas version
- Optimize with partitioning and caching

**Example Pattern:**
```python
from pyspark.sql import Window, functions as F

window = Window.partitionBy(group_cols).orderBy(time_col).rowsBetween(-window, 0)
df = df.withColumn("delta", F.col(value_col) - F.lag(value_col, 1).over(window))
```

---

#### 1.2 Spark Categorical Encoder

**Location:** `spark/features/categorical_encoder.py` (NEW)

**Required Methods:**
- `freq_encoding()` - Frequency-based encoding
- `target_mean_encoding()` - Smoothed target mean (with broadcast join)
- `rare_grouping()` - Group rare categories

**Key Requirements:**
- Use broadcast joins for small lookup tables
- Handle nulls and unseen categories
- Support sample weights (for compressed data)
- Cache intermediate results

**Example Pattern:**
```python
# Frequency encoding
freq_map = df.groupBy(cat_col).agg(F.count("*").alias("freq"))
df = df.join(broadcast(freq_map), cat_col)
df = df.withColumn(f"{cat_col}_freq", F.col("freq") / F.lit(total_count))
```

---

#### 1.3 Spark Feature Validator

**Location:** `spark/features/feature_validator.py` (NEW)

**Required Methods:**
- `validate_features()` - Compute metrics on Spark DataFrame
- `compute_psi()` - PSI calculation with Spark aggregations
- `to_dataframe()` - Convert results to pandas (small output)

**Key Requirements:**
- Use Spark aggregations for metrics (mean, std, min, max, quantiles)
- Handle large datasets efficiently
- Convert to pandas only for final results (small)
- Support sample weights

**Metrics to Compute:**
- missing_rate (count nulls / total)
- unique_count (approx_count_distinct)
- zero_variance (std == 0 check)
- min, max, mean, std (native Spark)
- p01, p99 (percentile_approx)
- skewness, kurtosis (requires UDF or collect)

---

### **Priority 2: Spark Optimization Utilities**

#### 2.1 Caching Manager

**Location:** `spark/optimization/caching.py` (NEW)

**Class:** `SparkCacheManager`

**Methods:**
- `cache_if_reused()` - Smart caching based on usage
- `persist_with_level()` - Persist with appropriate storage level
- `unpersist_all()` - Cleanup helper
- `get_cache_stats()` - Memory usage tracking

**Features:**
- Automatic cache level selection (MEMORY_ONLY vs MEMORY_AND_DISK)
- Track cache hits/misses
- Memory-aware caching (don't cache if too large)
- Integration with Spark UI

---

#### 2.2 Partition Optimizer

**Location:** `spark/optimization/partitioning.py` (NEW)

**Class:** `PartitionOptimizer`

**Methods:**
- `optimize_partitions()` - Repartition based on data size
- `coalesce_if_needed()` - Reduce partitions if too many
- `detect_skew()` - Identify skewed partitions
- `add_salting()` - Add salt for skewed keys
- `remove_salting()` - Remove salt after operations

**Features:**
- Automatic partition count calculation (target: 100-200MB per partition)
- Skew detection with configurable threshold
- Salting strategy for joins/groupBy on skewed keys
- Integration with AQE (Adaptive Query Execution)

---

#### 2.3 Query Optimizer

**Location:** `spark/optimization/query_optimizer.py` (NEW)

**Class:** `QueryOptimizer`

**Methods:**
- `optimize_join()` - Suggest broadcast join if applicable
- `prune_columns()` - Early column pruning
- `push_filters()` - Filter pushdown optimization
- `analyze_plan()` - Explain and suggest optimizations

**Features:**
- Broadcast join detection (small table < 10MB)
- Column pruning suggestions
- Filter pushdown analysis
- DAG visualization helpers

---

### **Priority 3: Spark Metrics Collection**

#### 3.1 Execution Metrics

**Location:** `spark/metrics/execution_metrics.py` (NEW)

**Class:** `SparkExecutionMetrics`

**Methods:**
- `collect_job_metrics()` - Collect Spark job metrics
- `get_stage_info()` - Stage-level statistics
- `get_task_metrics()` - Task-level metrics
- `get_memory_usage()` - Memory consumption
- `export_metrics()` - Export to JSON/CSV

**Metrics to Track:**
- Execution time per stage
- Shuffle read/write bytes
- Memory usage (on-heap, off-heap)
- Task count and duration
- Data size (input, output, shuffle)

---

#### 3.2 Performance Profiler

**Location:** `spark/metrics/performance_profiler.py` (NEW)

**Class:** `SparkPerformanceProfiler`

**Methods:**
- `profile_operation()` - Context manager for profiling
- `compare_plans()` - Compare execution plans
- `identify_bottlenecks()` - Find slow operations
- `suggest_optimizations()` - Auto-suggest improvements

**Features:**
- Automatic timing of operations
- Execution plan comparison
- Bottleneck detection (slow stages, large shuffles)
- Optimization suggestions

---

## Implementation Plan

### **Phase 1: Enhanced Features Spark Support** (High Priority)

1. **Create `spark/features/` directory structure:**
   ```
   spark/features/
   ├── __init__.py
   ├── temporal_trends.py
   ├── categorical_encoder.py
   └── feature_validator.py
   ```

2. **Implement Spark versions:**
   - Start with `temporal_trends.py` (most critical)
   - Then `categorical_encoder.py` (frequently used)
   - Finally `feature_validator.py` (validation)

3. **Testing:**
   - Unit tests with small Spark DataFrames
   - Integration tests comparing pandas vs Spark results
   - Performance benchmarks

---

### **Phase 2: Optimization Utilities** (Medium Priority)

1. **Create `spark/optimization/` implementations:**
   - `caching.py` - Cache management
   - `partitioning.py` - Partition optimization
   - `query_optimizer.py` - Query analysis

2. **Integration:**
   - Integrate with `SparkFeatureEngineer`
   - Add automatic optimization hooks
   - Config-driven optimization levels

---

### **Phase 3: Metrics Collection** (Lower Priority)

1. **Create `spark/metrics/` implementations:**
   - `execution_metrics.py` - Job/stage metrics
   - `performance_profiler.py` - Profiling tools

2. **Integration:**
   - Add to pipeline for monitoring
   - Export metrics for analysis
   - Dashboard integration

---

## Technical Considerations

### **Spark Window Functions**

For temporal features, use Window functions:
```python
from pyspark.sql import Window
window = Window.partitionBy("customer_id").orderBy("date").rowsBetween(-6, 0)
df = df.withColumn("rolling_mean", F.avg("balance").over(window))
```

### **Broadcast Joins**

For categorical encoding lookups:
```python
from pyspark.sql.functions import broadcast
small_df = freq_map  # < 10MB
df = df.join(broadcast(small_df), "category")
```

### **Sample Weights**

Support compressed data with weights:
```python
# Weighted aggregation
df.groupBy("bin").agg(
    F.sum("sample_weight").alias("total_weight"),
    F.sum(F.col("target") * F.col("sample_weight")).alias("weighted_events")
)
```

### **Null Handling**

Consistent null handling across pandas and Spark:
```python
# Spark null handling
df = df.withColumn("value", F.coalesce(F.col("value"), F.lit(0)))
```

### **Performance Optimization**

1. **Caching Strategy:**
   - Cache DataFrames used multiple times
   - Use MEMORY_AND_DISK for large DataFrames
   - Unpersist after use

2. **Partitioning:**
   - Repartition before expensive operations
   - Coalesce after filtering
   - Use same partitioner for joins

3. **Query Optimization:**
   - Prune columns early
   - Push filters down
   - Use broadcast joins for small tables

---

## Testing Strategy

### **Unit Tests:**
- Test each Spark method independently
- Use small test DataFrames
- Verify null handling
- Test edge cases (empty, single row, all nulls)

### **Integration Tests:**
- Compare pandas vs Spark results (small data)
- Verify equivalence within tolerance
- Test with sample weights
- Test with compressed data

### **Performance Tests:**
- Benchmark on large datasets (10M+ rows)
- Measure execution time
- Compare with pandas baseline
- Track memory usage

---

## Documentation Requirements

1. **API Documentation:**
   - Sphinx docs for all new classes
   - Usage examples
   - Performance notes

2. **User Guide:**
   - When to use Spark vs pandas
   - Optimization best practices
   - Troubleshooting guide

3. **Jupyter Playbook:**
   - `playbooks/10_spark_enhanced_features.ipynb`
   - Demonstrate Spark temporal features
   - Show optimization utilities
   - Performance comparisons

---

## Dependencies

### **Required:**
- `pyspark>=3.4.0` (already in requirements)
- `numpy` (for statistical operations)
- `scipy` (for trend_slope regression)

### **Optional:**
- `koalas` (pandas-like API on Spark) - consider for easier migration

---

## Success Criteria

1. ✅ All enhanced features work with Spark DataFrames
2. ✅ Performance improvements vs pandas (10x+ for large data)
3. ✅ Results match pandas version (within tolerance)
4. ✅ Optimization utilities reduce execution time
5. ✅ Metrics collection provides actionable insights
6. ✅ Complete test coverage (>80%)
7. ✅ Documentation complete with examples

---

## Estimated Effort

- **Phase 1 (Enhanced Features):** 3-4 days
- **Phase 2 (Optimization):** 2-3 days
- **Phase 3 (Metrics):** 1-2 days
- **Testing & Documentation:** 2-3 days

**Total:** ~8-12 days

---

## Next Steps

1. **Review and approve this assessment**
2. **Prioritize phases** (recommend: Phase 1 first)
3. **Create implementation plan** with detailed tasks
4. **Start with temporal trends** (highest value)

---

**Status:** Assessment Complete - Ready for Implementation Planning
