Spark Layer
===========

Spark session management, compression, optimization utilities, and metrics collection.

Session Management
------------------

.. automodule:: cr_score.spark.session
   :members:
   :undoc-members:
   :show-inheritance:

Compression
-----------

.. automodule:: cr_score.spark.compression.post_binning_exact
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Utilities
----------------------

Caching Manager
~~~~~~~~~~~~~~~

Intelligent caching with automatic storage level selection.

.. autoclass:: cr_score.spark.optimization.caching.SparkCacheManager
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from cr_score.spark import SparkCacheManager
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()
   cache_mgr = SparkCacheManager(spark)

   # Cache DataFrame if it will be reused
   df_cached = cache_mgr.cache_if_reused(df, "intermediate_result", min_reuses=2)

   # Get cache statistics
   stats = cache_mgr.get_cache_stats()

   # Cleanup
   cache_mgr.unpersist("intermediate_result")

Partition Optimizer
~~~~~~~~~~~~~~~~~~

Partition optimization, skew detection, and salting strategies.

.. autoclass:: cr_score.spark.optimization.partitioning.PartitionOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from cr_score.spark import PartitionOptimizer

   optimizer = PartitionOptimizer(spark)

   # Optimize partition count
   df_optimized = optimizer.optimize_partitions(df)

   # Detect skew
   skew_info = optimizer.detect_skew(df, "customer_id")

   # Add salting for skewed joins
   df_salted, salted_col = optimizer.add_salting(df, "customer_id", num_salts=10)

Metrics Collection
------------------

Execution Metrics
~~~~~~~~~~~~~~~~~

Collect job, stage, and task-level metrics.

.. autoclass:: cr_score.spark.metrics.execution_metrics.SparkExecutionMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from cr_score.spark import SparkExecutionMetrics

   metrics = SparkExecutionMetrics(spark)

   # Track an operation
   result, metrics_data = metrics.track_execution(
       "feature_engineering",
       engineer.fit_transform,
       df
   )

   # Get executor metrics
   executors = metrics.get_executor_metrics()

   # Export metrics
   metrics.export_metrics("metrics.json")

Performance Profiler
~~~~~~~~~~~~~~~~~~~~

Performance profiling and bottleneck analysis.

.. autoclass:: cr_score.spark.metrics.performance_profiler.PerformanceProfiler
   :members:
   :undoc-members:
   :show-inheritance:

Example:

.. code-block:: python

   from cr_score.spark import PerformanceProfiler

   profiler = PerformanceProfiler(spark)

   # Profile an operation
   profile = profiler.profile_operation(
       "feature_engineering",
       engineer.fit_transform,
       df
   )

   # Analyze bottlenecks
   bottlenecks = profiler.analyze_bottlenecks(profile)

   # Generate report
   report = profiler.generate_report()
   print(report)
