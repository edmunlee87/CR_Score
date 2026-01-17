Spark Optimization & Metrics
=============================

Guide to using Spark optimization utilities and metrics collection for performance tuning.

Overview
--------

The Spark optimization and metrics modules provide tools for:

* **Intelligent Caching**: Automatic cache level selection and memory management
* **Partition Optimization**: Skew detection, salting, and partition count optimization
* **Performance Metrics**: Job, stage, and task-level metrics collection
* **Performance Profiling**: Bottleneck analysis and optimization suggestions

Caching Management
-----------------

Use ``SparkCacheManager`` for intelligent caching of intermediate results.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.spark import SparkCacheManager
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()
   cache_mgr = SparkCacheManager(spark)

   # Cache DataFrame if it will be reused
   df_cached = cache_mgr.cache_if_reused(
       df_intermediate,
       "features",
       min_reuses=2
   )

   # Use cached DataFrame multiple times
   result1 = df_cached.groupBy("customer_id").agg(...)
   result2 = df_cached.join(other_df, "customer_id")

   # Get cache statistics
   stats = cache_mgr.get_cache_stats()
   print(f"Cache hits: {stats['features']['hits']}")

   # Cleanup when done
   cache_mgr.unpersist("features")

Automatic Cache Level Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cache manager automatically selects the optimal storage level:

.. code-block:: python

   # Small DataFrame: MEMORY_ONLY
   df_small = spark.createDataFrame(...)  # < 1M rows
   df_cached = cache_mgr.persist_with_level(df_small, "small_df")
   # Uses MEMORY_ONLY

   # Large DataFrame: MEMORY_AND_DISK
   df_large = spark.read.parquet("large_data.parquet")  # > 10M rows
   df_cached = cache_mgr.persist_with_level(df_large, "large_df")
   # Uses MEMORY_AND_DISK (spills to disk if needed)

Partition Optimization
----------------------

Use ``PartitionOptimizer`` to optimize partition counts and handle skew.

Optimize Partition Count
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.spark import PartitionOptimizer

   optimizer = PartitionOptimizer(spark, target_partition_size_mb=128.0)

   # Optimize based on data size (target: 128MB per partition)
   df_optimized = optimizer.optimize_partitions(df)

   print(f"Partitions: {df_optimized.rdd.getNumPartitions()}")

Detect Skew
~~~~~~~~~~~

.. code-block:: python

   # Detect skew in a key column
   skew_info = optimizer.detect_skew(df, "customer_id", sample_fraction=0.1)

   if skew_info["skewed"]:
       print(f"Skew detected: max/mean ratio = {skew_info['ratio']:.2f}")
       # Add salting to mitigate skew
       df_salted, salted_col = optimizer.add_salting(
           df,
           "customer_id",
           num_salts=10
       )
   else:
       print("No significant skew detected")

Handle Skewed Joins
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add salting before join
   df1_salted, salted_col = optimizer.add_salting(df1, "customer_id", num_salts=10)
   df2_salted, _ = optimizer.add_salting(df2, "customer_id", num_salts=10)

   # Join on salted column
   df_joined = df1_salted.join(
       df2_salted,
       df1_salted[salted_col] == df2_salted[salted_col]
   )

   # Remove salt after join
   df_final = optimizer.remove_salting(df_joined, salted_col, "customer_id")

Coalesce Too Many Partitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If you have too many partitions (e.g., after filtering)
   df_filtered = df.filter("date > '2024-01-01'")
   df_coalesced = optimizer.coalesce_if_needed(df_filtered, max_partitions=200)

Performance Metrics
-------------------

Use ``SparkExecutionMetrics`` to collect job and executor metrics.

Track Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.spark import SparkExecutionMetrics

   metrics = SparkExecutionMetrics(spark)

   # Track an operation
   result, metrics_data = metrics.track_execution(
       "feature_engineering",
       engineer.fit_transform,
       df
   )

   print(f"Duration: {metrics_data['duration_seconds']:.2f}s")
   print(f"Status: {metrics_data['status']}")

Get Executor Metrics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   executors = metrics.get_executor_metrics()

   for executor in executors:
       print(f"Executor {executor['executor_id']}:")
       print(f"  Active tasks: {executor['active_tasks']}")
       print(f"  Completed tasks: {executor['completed_tasks']}")
       print(f"  Memory: {executor['max_memory'] / 1024**3:.2f} GB")

Export Metrics
~~~~~~~~~~~~~~

.. code-block:: python

   # Export all metrics to JSON
   metrics.export_metrics("spark_metrics.json")

   # Get execution history
   history = metrics.get_metrics_history()
   for entry in history:
       print(f"{entry['operation']}: {entry['duration_seconds']:.2f}s")

Performance Profiling
---------------------

Use ``PerformanceProfiler`` to analyze performance and identify bottlenecks.

Profile Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.spark import PerformanceProfiler

   profiler = PerformanceProfiler(spark)

   # Profile a single operation
   profile = profiler.profile_operation(
       "feature_engineering",
       engineer.fit_transform,
       df
   )

   print(f"Duration: {profile['duration_seconds']:.2f}s")
   print(f"Partitions: {profile['dataframe_metrics']['num_partitions']}")

Analyze Bottlenecks
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bottlenecks = profiler.analyze_bottlenecks(profile)

   for suggestion in bottlenecks:
       print(f"- {suggestion}")

Compare Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   operations = {
       "method1": method1,
       "method2": method2,
   }

   results = profiler.compare_operations(operations, df)

   print(f"Fastest: {results['_comparison']['fastest']}")
   print(f"Speedup: {results['_comparison']['speedup']:.2f}x")

Generate Report
~~~~~~~~~~~~~~~

.. code-block:: python

   # Profile multiple operations
   profiler.profile_operation("op1", func1, df)
   profiler.profile_operation("op2", func2, df)

   # Generate comprehensive report
   report = profiler.generate_report()
   print(report)

Complete Workflow Example
-------------------------

.. code-block:: python

   from cr_score.spark import (
       SparkCacheManager,
       PartitionOptimizer,
       PerformanceProfiler,
       SparkExecutionMetrics,
   )
   from cr_score.features import FeatureEngineer

   # Initialize utilities
   spark = SparkSession.builder.getOrCreate()
   cache_mgr = SparkCacheManager(spark)
   optimizer = PartitionOptimizer(spark)
   profiler = PerformanceProfiler(spark)
   metrics = SparkExecutionMetrics(spark)

   # Load and optimize data
   df = spark.read.parquet("data.parquet")
   df = optimizer.optimize_partitions(df)

   # Check for skew
   skew_info = optimizer.detect_skew(df, "customer_id")
   if skew_info["skewed"]:
       df, salted_col = optimizer.add_salting(df, "customer_id")

   # Profile feature engineering
   config = FeatureEngineeringConfig(...)
   engineer = FeatureEngineer(config)

   profile = profiler.profile_operation(
       "feature_engineering",
       engineer.fit_transform,
       df
   )

   # Cache intermediate results
   df_features = cache_mgr.cache_if_reused(
       profile["result"],
       "features",
       min_reuses=2
   )

   # Analyze performance
   bottlenecks = profiler.analyze_bottlenecks(profile)
   print(profiler.generate_report())

   # Export metrics
   metrics.export_metrics("metrics.json")

   # Cleanup
   cache_mgr.unpersist("features")

Best Practices
--------------

1. **Caching**: Only cache DataFrames that are reused multiple times
2. **Partitioning**: Target 100-200MB per partition for optimal performance
3. **Skew**: Use salting for joins/groupBy on skewed keys (>3x ratio)
4. **Metrics**: Track operations in production for performance monitoring
5. **Profiling**: Profile expensive operations to identify bottlenecks
