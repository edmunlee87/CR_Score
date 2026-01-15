================================================================================
CR_Score SPARK OPERATIONS GUIDE
================================================================================

Version: 1.0
Date: 2026-01-15
Purpose: Define Spark-specific patterns, optimizations, and correctness rules

================================================================================
1. SPARK SESSION MANAGEMENT
================================================================================

Session Creation Pattern:

    from CR_Score.spark.session import get_or_create_spark_session
    
    def get_spark_session(config: ExecutionConfig) -> "pyspark.sql.SparkSession":
        """Get or create Spark session with config."""
        return get_or_create_spark_session(
            app_name="CR_Score-binning",
            engine=config.execution.engine,  # "spark_local", "spark_cluster"
            config_dict={
                "spark.sql.shuffle.partitions": config.execution.spark.shuffle_partitions,
                "spark.sql.adaptive.enabled": True,
                "spark.sql.adaptive.coalescePartitions.enabled": True,
                "spark.memory.storageFraction": 0.3,
                "spark.sql.files.maxPartitionBytes": "128mb",
            }
        )

Session Configuration:

    LOCAL MODE (spark_local):
    - spark.master = "local[*]"
    - spark.driver.memory = "4g"
    - Use for development and testing
    
    CLUSTER MODE (spark_cluster):
    - spark.master = "yarn" or "k8s://..."
    - Configured via config.execution.spark section
    - Production deployments only
    - Requires cluster credentials in environment

Shutdown Pattern:

    def cleanup_spark_session(spark: "pyspark.sql.SparkSession"):
        """Properly shutdown Spark session."""
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

================================================================================
2. DATAFRAME OPERATIONS
================================================================================

Import Pattern:

    from pyspark.sql import DataFrame
    from pyspark.sql.functions import (
        col, when, sum as F_sum, count, avg, max as F_max,
        rand, concat, lit, round, lower, trim, coalesce,
    )

Type-Safe Column Selection:

    # Good: Explicit column selection
    selected_df = df.select("age", "income", "target")
    
    # Good: Using col() for transformations
    result_df = df.select(col("age"), F_max("income").over(...))
    
    # Bad: String references without col()
    result_df = df.select("age").filter("age > 30")  # Can fail in complex DAGs
    
    # Better:
    result_df = df.select(col("age")).filter(col("age") > 30)

Column Naming Conventions:

    - Source columns: lowercase_underscore (age, home_value)
    - Bin columns: {variable}_bin (age_bin, income_bin)
    - WoE columns: {variable}_woe (age_woe, income_woe)
    - Weight columns: sample_weight, event_weight, row_weight
    - Internal temp: _temp_<name> (prefix with underscore)

NULL Handling:

    # Explicit null checks
    df_clean = (
        df
        .filter(col("age").isNotNull())
        .filter(col("target").isin([0, 1]))
    )
    
    # Fill nulls with value
    df_filled = df.fillna({"age": -999, "income": 0})
    
    # Fill with forward fill (sorted)
    from pyspark.sql.window import Window
    window = Window.partitionBy("customer_id").orderBy("month")
    df_ffill = df.withColumn("age", F.last("age", ignoredNulls=True).over(window))

DataFrame Validation:

    def validate_dataframe(df: DataFrame, required_cols: List[str]):
        """Validate DataFrame structure."""
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing columns: {missing_cols}")
        
        row_count = df.count()  # Trigger evaluation
        if row_count == 0:
            raise DataValidationError("DataFrame is empty")
        
        logger.info(f"DataFrame validated: {row_count} rows, {len(df.columns)} cols")

================================================================================
3. AGGREGATION WITH SAMPLE WEIGHTS
================================================================================

Weighted Aggregation Pattern:

    def aggregate_weighted(
        df: DataFrame,
        group_cols: List[str],
        weight_col: str = "sample_weight",
    ) -> DataFrame:
        """Aggregate with weights (e.g., for compressed data)."""
        from pyspark.sql.functions import sum as F_sum
        
        agg_df = (
            df
            .groupBy(*group_cols)
            .agg(
                F_sum(weight_col).alias("total_weight"),
                F_sum(col("target") * col(weight_col)).alias("total_events"),
                (F_sum(col("target") * col(weight_col)) / 
                 F_sum(weight_col)).alias("event_rate")
            )
        )
        return agg_df

Post-Binning Exact Compression:

    def compress_exact(
        df: DataFrame,
        bin_cols: List[str],
        segment_cols: List[str],
    ) -> DataFrame:
        """Compress via post-binning aggregation (preserves likelihood exactly)."""
        group_cols = bin_cols + segment_cols
        
        compressed = (
            df
            .groupBy(*group_cols)
            .agg(
                count("*").alias("sample_weight"),
                F_sum(col("target")).alias("event_weight")
            )
            .withColumn(
                "event_rate",
                col("event_weight") / col("sample_weight")
            )
        )
        
        # Verify totals preserved
        original_rows = df.count()
        original_events = df.agg(F_sum("target")).collect()[0][0]
        
        compressed_rows = compressed.agg(F_sum("sample_weight")).collect()[0][0]
        compressed_events = compressed.agg(F_sum("event_weight")).collect()[0][0]
        
        assert original_rows == compressed_rows, "Row count mismatch"
        assert original_events == compressed_events, "Event count mismatch"
        
        return compressed

Weighted Logistic Regression Integration:

    def train_weighted_logit(
        df: DataFrame,
        feature_cols: List[str],
        sample_weight_col: str = "sample_weight",
    ):
        """Train logistic regression with sample weights."""
        # Convert to Pandas (small enough after compression)
        pdf = df.select([*feature_cols, sample_weight_col, "event_rate"]).toPandas()
        
        X = pdf[feature_cols]
        y = pdf["event_rate"]  # Already aggregated
        weights = pdf[sample_weight_col]
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y, sample_weight=weights)
        
        return model

================================================================================
4. SKEW DETECTION & HANDLING
================================================================================

Skew Detection:

    def detect_skew(df: DataFrame, key_col: str, threshold: float = 0.1):
        """Detect if key column is skewed."""
        from pyspark.sql.functions import count, max as F_max, avg
        
        partition_stats = (
            df
            .groupBy(key_col)
            .agg(count("*").alias("partition_size"))
            .agg(
                F_max("partition_size").alias("max_size"),
                avg("partition_size").alias("avg_size")
            )
            .collect()[0]
        )
        
        max_size = partition_stats["max_size"]
        avg_size = partition_stats["avg_size"]
        skew_ratio = max_size / avg_size if avg_size > 0 else 0
        
        is_skewed = skew_ratio > (1 + threshold)
        logger.info(f"Skew detection: ratio={skew_ratio:.2f}, skewed={is_skewed}")
        
        return is_skewed, skew_ratio

Salting Strategy:

    def add_salting(
        df: DataFrame,
        key_col: str,
        salt_factor: int = 8,
        seed: int = 42,
    ) -> DataFrame:
        """Add salt to skewed column for load balancing."""
        from pyspark.sql.functions import rand, cast
        from pyspark.sql.types import IntegerType
        
        df_salted = (
            df
            .withColumn(
                "_salt",
                cast(rand(seed) * salt_factor, IntegerType())
            )
            .withColumn(
                key_col,
                concat(col(key_col), lit("_"), col("_salt"))
            )
        )
        return df_salted

    def remove_salting(df: DataFrame, key_col: str) -> DataFrame:
        """Remove salt after join/groupby (reverse operation)."""
        # Assuming salt format: "original_value_N"
        df_clean = df.withColumn(
            key_col,
            concat(
                col(key_col).substr(1, col(key_col).length() - 2)
            )
        )
        return df_clean

================================================================================
5. PERSISTENCE & CHECKPOINTING
================================================================================

When to Persist:

    Use persist() when DataFrame is used MULTIPLE times:
    - Reused in multiple aggregations
    - Parent of multiple downstream paths
    - Expensive to recompute (complex joins, UDFs)
    
    Don't persist:
    - Single-use DataFrames
    - Already persisted upstream
    - Very large DataFrames (use checkpoint instead)

Persistence Pattern:

    from pyspark import StorageLevel
    
    df_intermediate = expensive_computation(df)
    df_intermediate.persist(StorageLevel.MEMORY_AND_DISK)
    
    result_1 = df_intermediate.groupBy(...).agg(...)
    result_2 = df_intermediate.join(...).select(...)
    
    df_intermediate.unpersist()  # Release memory after use

Checkpointing:

    Checkpoint breaks the DAG chain:
    - Saves intermediate result to disk
    - Used for long DAGs (>10 stages)
    - Prevents recomputation if earlier stage fails
    
    Pattern:
    
    if df.rdd.getNumPartitions() > 800 or df.rdd.toDebugString().count("\n") > 100:
        checkpoint_dir = config.execution.spark.checkpoint_dir
        spark.sparkContext.setCheckpointDir(checkpoint_dir)
        df = df.checkpoint()  # Materialize and save
        logger.info(f"Checkpoint created: {checkpoint_dir}")

================================================================================
6. PARTITION OPTIMIZATION
================================================================================

Setting Shuffle Partitions:

    # Global config (via config.yml)
    spark.sql.shuffle.partitions = 800  # Default for 100M+ rows
    
    # Adjust for data size:
    - Small data (< 1M rows):  50 partitions
    - Medium (1M - 100M rows): 200 partitions
    - Large (> 100M rows):     800 partitions
    - Huge (> 1B rows):        2000+ partitions
    
    # Rule: 1 partition = ~100-200MB of data

Explicit Repartitioning:

    def optimize_partitions(df: DataFrame, target_rows: int) -> DataFrame:
        """Optimize partitions based on target row count."""
        num_partitions = max(1, target_rows // (100 * 1024 * 1024))  # 100MB per partition
        return df.repartition(num_partitions)

Partitioning for Joins:

    # Pre-sort both DataFrames on join key
    df_a = df_a.sort("customer_id")
    df_b = df_b.sort("customer_id")
    
    # Use same number of partitions
    df_a = df_a.repartition(800, "customer_id")
    df_b = df_b.repartition(800, "customer_id")
    
    result = df_a.join(df_b, "customer_id")

================================================================================
7. CORRECTNESS GUARANTEES
================================================================================

Numeric Precision:

    # Use DECIMAL for financial data
    from pyspark.sql.types import DecimalType
    df = df.withColumn("amount", col("amount").cast(DecimalType(15, 2)))
    
    # Ensure consistent rounding
    def safe_sum_check(df: DataFrame, col_name: str, expected_sum: float):
        """Verify sum matches expected with tolerance."""
        actual_sum = df.agg(F_sum(col_name)).collect()[0][0]
        relative_error = abs(actual_sum - expected_sum) / expected_sum
        if relative_error > 1e-6:
            raise ValueError(f"Sum mismatch: {actual_sum} vs {expected_sum}")

Equivalence Testing Pattern:

    def test_spark_vs_pandas_equivalence():
        """Verify Spark implementation matches pandas for small data."""
        # Small dataset (e.g., 10K rows)
        pdf = pd.read_csv("small_sample.csv")
        sdf = spark.createDataFrame(pdf)
        
        # Run operation both ways
        result_pandas = operation_pandas(pdf)
        result_spark = operation_spark(sdf).toPandas()
        
        # Compare with tolerance
        pd.testing.assert_frame_equal(
            result_pandas.sort_values("id").reset_index(drop=True),
            result_spark.sort_values("id").reset_index(drop=True),
            check_dtype=False,
            atol=1e-10
        )

Determinism Verification:

    def verify_determinism(df: DataFrame, operation, runs: int = 3):
        """Run operation multiple times, verify identical results."""
        results = []
        for i in range(runs):
            result = operation(df)
            content_hash = compute_hash(result.toPandas().to_json())
            results.append(content_hash)
        
        if len(set(results)) > 1:
            raise ValueError(f"Non-deterministic results: {results}")
        
        logger.info(f"Determinism verified ({runs} runs)")

================================================================================
8. PERFORMANCE OPTIMIZATION
================================================================================

Query Optimization Tips:

    ☐ Prune columns early (select only needed columns)
    ☐ Filter early (filter before join/groupby)
    ☐ Use broadcast for small DataFrames in joins:
    
        from pyspark.sql.functions import broadcast
        result = large_df.join(broadcast(small_df), "key")
    
    ☐ Use SQL for complex queries (better optimizer)
    ☐ Avoid UDFs (use native functions)
    ☐ Cache frequently-accessed DataFrames

Memory Management:

    - Set executor memory in config: spark.executor.memory = "2g"
    - Monitor with: df.count() (forces materialization)
    - Use df.explain(True) to inspect execution plan
    - Set spark.memory.storageFraction = 0.3 (60% for computation, 30% for cache)

Monitoring DAG:

    df.explain(True)  # Print optimized execution plan
    
    Look for:
    - Unnecessary shuffles (reduce by partitioning strategy)
    - Broadcast joins (good)
    - Filter pushdowns (good)
    - Full table scans (avoid if possible)

================================================================================
9. SPARK DATAFRAME SCHEMA MANAGEMENT
================================================================================

Schema Definition:

    from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
    
    schema = StructType([
        StructField("customer_id", IntegerType(), False),  # Not null
        StructField("age", IntegerType(), True),           # Nullable
        StructField("income", DoubleType(), True),
        StructField("target", IntegerType(), True),
    ])
    
    df = spark.read.schema(schema).csv("data.csv")

Schema Inference (Use with Caution):

    # Automatic but slow for large files
    df = spark.read.option("inferSchema", True).csv("data.csv")
    
    # Better: Provide schema explicitly or sample first
    df_sample = spark.read.csv("data.csv", header=True).limit(1000)
    schema = df_sample.schema
    df = spark.read.schema(schema).csv("data.csv")

Schema Validation:

    def validate_schema(df: DataFrame, expected_schema: StructType):
        """Validate DataFrame schema matches expectation."""
        if df.schema != expected_schema:
            raise SchemaError(f"Schema mismatch:\n{df.schema}\nvs\n{expected_schema}")

================================================================================
10. SPARK-PYTHON INTEROP
================================================================================

Efficient Data Transfer:

    # Small results OK to collect
    results = df.collect()  # Brings to driver, OK for < 100K rows
    
    # For larger results, use toPandas with partition limit
    pdf = df.limit(1000000).toPandas()
    
    # For huge results, use Pandas UDF (distributed execution)
    from pyspark.sql.functions import pandas_udf
    
    @pandas_udf(DoubleType())
    def compute_score(batch_pdf):
        return batch_pdf.apply(scoring_function)
    
    df_scored = df.withColumn("score", compute_score(col("features")))

Arrow Optimization:

    # Enable Arrow for faster transfer
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    pdf = df.toPandas()  # Much faster with Arrow

UDF Patterns:

    # Use Pandas UDF (preferred, distributed)
    @pandas_udf(IntegerType())
    def bin_age(ages: pd.Series) -> pd.Series:
        return pd.cut(ages, bins=[0, 30, 60, 100], labels=[1, 2, 3])
    
    df_binned = df.withColumn("age_bin", bin_age(col("age")))
    
    # NOT: Python UDF (slower, single-threaded)
    @udf(IntegerType())
    def slow_bin(age: int) -> int:
        if age < 30: return 1
        ...
    df_binned = df.withColumn("age_bin", slow_bin(col("age")))

================================================================================
11. ERROR HANDLING IN SPARK
================================================================================

Task Failures:

    # Spark automatically retries tasks (default: 4 retries)
    # Configure via: spark.task.maxFailures = 10
    
    # For expected failures, catch and log
    try:
        result = df.collect()
    except Exception as e:
        logger.error(f"Spark job failed: {e}")
        raise CR_ScoreException(f"Processing failed: {e}") from e

Executor OOM:

    # Increase spark.executor.memory in config
    # Reduce partition size: spark.sql.shuffle.partitions
    # Use compression: spark.shuffle.compress = true

Data Skew Failures:

    # Use salting + filtering to reduce partition size
    df_salted = add_salting(df, "customer_id", salt_factor=8)
    # ... process ...
    df_cleaned = remove_salting(df_salted, "customer_id")

================================================================================
