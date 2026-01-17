Enhanced Feature Engineering
============================

The enhanced feature engineering module provides advanced capabilities for creating temporal, categorical, and validated features for credit risk modeling.

Overview
--------

The enhanced features include:

* **Temporal Trend Features**: Delta, percent change, momentum, volatility, trend slope, rolling rank
* **Categorical Encoding**: Frequency encoding, target mean encoding, rare grouping
* **Feature Validation**: Quality metrics, thresholds, PSI calculation
* **Dependency Management**: Graph-based dependency resolution with cycle detection
* **Feature Registry**: Metadata tracking and lineage for audit trails

Temporal Trend Features
-----------------------

Create time-based features for capturing trends and patterns.

**Spark Support:** Automatically detects Spark DataFrames and uses Spark Window functions for distributed processing.

.. autoclass:: cr_score.features.enhanced_features.TemporalTrendFeatures
   :members:
   :undoc-members:
   :show-inheritance:

Example with Spark:
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.features import TemporalTrendFeatures
   from pyspark.sql import SparkSession

   spark = SparkSession.builder.getOrCreate()
   df_spark = spark.read.parquet("data.parquet")

   trend = TemporalTrendFeatures()
   # Automatically uses Spark implementation
   df_delta = trend.delta(df_spark, "balance", time_col="date", group_cols=["customer_id"])
   df_momentum = trend.momentum(df_spark, "balance", window=3, group_cols=["customer_id"])

Delta
~~~~~

Compute the difference between consecutive periods.

.. code-block:: python

   from cr_score.features import TemporalTrendFeatures
   
   trend = TemporalTrendFeatures()
   df = trend.delta(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id'],
       periods=1
   )

Percent Change
~~~~~~~~~~~~~~

Compute the percentage change from previous periods.

.. code-block:: python

   df = trend.pct_change(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id']
   )

Momentum
~~~~~~~~

Calculate momentum as the difference between current value and rolling mean.

.. code-block:: python

   df = trend.momentum(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id'],
       window=3
   )

Volatility
~~~~~~~~~~

Measure volatility using rolling standard deviation or coefficient of variation.

.. code-block:: python

   df = trend.volatility(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id'],
       window=6,
       method='std'  # or 'cv' for coefficient of variation
   )

Trend Slope
~~~~~~~~~~~

Compute trend slope using linear regression over rolling window.

.. code-block:: python

   df = trend.trend_slope(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id'],
       window=6
   )

Rolling Rank
~~~~~~~~~~~~

Calculate the rank of the current value within a rolling window.

.. code-block:: python

   df = trend.rolling_rank(
       df,
       column='balance',
       time_col='date',
       group_cols=['customer_id'],
       window=6,
       pct=True  # Return percentile rank (0-1)
   )

Categorical Encoding
-------------------

Encode categorical variables for modeling.

**Spark Support:** Automatically detects Spark DataFrames and uses broadcast joins for efficient distributed processing.

.. autoclass:: cr_score.features.enhanced_features.CategoricalEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Frequency Encoding
~~~~~~~~~~~~~~~~~~

Encode categories by their frequency in the dataset.

.. code-block:: python

   from cr_score.features import CategoricalEncoder
   
   encoder = CategoricalEncoder()
   # Works with both pandas and Spark DataFrames
   df = encoder.freq_encoding(df, 'account_type')
   
   # With Spark DataFrame (automatic)
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.getOrCreate()
   df_spark = spark.createDataFrame(df)
   df_encoded = encoder.freq_encoding(df_spark, 'account_type')  # Uses Spark automatically

Target Mean Encoding
~~~~~~~~~~~~~~~~~~~~

Encode categories using smoothed target means.

.. code-block:: python

   df = encoder.target_mean_encoding(
       df,
       column='account_type',
       target='is_default',
       smoothing=5.0,
       min_samples_leaf=10
   )

Rare Category Grouping
~~~~~~~~~~~~~~~~~~~~~~

Group infrequent categories together.

.. code-block:: python

   df = encoder.rare_grouping(
       df,
       column='region',
       threshold=0.05,  # 5% frequency threshold
       rare_label='RARE'
   )

Feature Validation
------------------

Validate features and compute quality metrics.

.. autoclass:: cr_score.features.enhanced_features.FeatureValidator
   :members:
   :undoc-members:
   :show-inheritance:

Basic Validation
~~~~~~~~~~~~~~~~

Validate features and generate quality reports.

.. code-block:: python

   from cr_score.features import FeatureValidator
   
   validator = FeatureValidator(
       warning_thresholds={'missing_rate': 0.05},
       hard_fail_thresholds={'missing_rate': 0.20}
   )
   
   results = validator.validate_features(df, feature_list=['balance', 'dpd'])

Available Metrics
~~~~~~~~~~~~~~~~~

* **missing_rate**: Proportion of missing values
* **unique_count**: Number of unique values
* **zero_variance**: Whether feature has zero variance
* **min, max, mean, std**: Statistical metrics
* **p01, p99**: 1st and 99th percentiles
* **skewness, kurtosis**: Distribution shape metrics

PSI Calculation
~~~~~~~~~~~~~~~

Compute Population Stability Index for distribution drift.

.. code-block:: python

   psi = validator.compute_psi(
       baseline_dist=baseline_series,
       current_dist=current_series,
       bins=10
   )
   
   if psi > 0.25:
       print("Significant distribution shift detected!")

Dependency Management
--------------------

Manage feature dependencies and execution order.

.. autoclass:: cr_score.features.enhanced_features.DependencyGraph
   :members:
   :undoc-members:
   :show-inheritance:

Building Dependency Graph
~~~~~~~~~~~~~~~~~~~~~~~~~

Define features and their dependencies.

.. code-block:: python

   from cr_score.features import DependencyGraph
   
   graph = DependencyGraph()
   
   graph.add_feature('balance', [])
   graph.add_feature('credit_limit', [])
   graph.add_feature('utilization', ['balance', 'credit_limit'])
   graph.add_feature('log_utilization', ['utilization'])

Topological Sort
~~~~~~~~~~~~~~~~

Get execution order that respects dependencies.

.. code-block:: python

   execution_order = graph.topological_sort()
   # Returns: ['balance', 'credit_limit', 'utilization', 'log_utilization']

Cycle Detection
~~~~~~~~~~~~~~~

Detect circular dependencies.

.. code-block:: python

   try:
       execution_order = graph.topological_sort()
   except ValueError as e:
       print(f"Cycle detected: {e}")

Feature Registry
----------------

Track feature metadata and lineage for audit purposes.

.. autoclass:: cr_score.features.engineering.FeatureRegistry
   :members:
   :undoc-members:
   :show-inheritance:

Registering Features
~~~~~~~~~~~~~~~~~~~~

Register features with metadata.

.. code-block:: python

   from cr_score.features import FeatureRegistry
   
   registry = FeatureRegistry()
   
   registry.register(
       name='utilization',
       source_columns=['balance', 'credit_limit'],
       operation='ratio',
       parameters={},
       window=None,
       missing_strategy='zero',
       dependencies=[],
       engine='pandas',
       output_dtype='float64'
   )

Exporting Registry
~~~~~~~~~~~~~~~~~~

Export registry for audit trails.

.. code-block:: python

   # Export to JSON
   registry.export_json('feature_registry.json')
   
   # Export to dictionary
   registry_dict = registry.export_dict()

Feature Lineage
~~~~~~~~~~~~~~~

Get full lineage including transitive dependencies.

.. code-block:: python

   lineage = registry.get_lineage('log_utilization')
   # Returns dependency tree with all sources and operations

Complete Example
----------------

Putting it all together:

.. code-block:: python

   import pandas as pd
   from cr_score.features import (
       TemporalTrendFeatures,
       CategoricalEncoder,
       FeatureValidator,
       DependencyGraph,
       FeatureRegistry,
   )
   
   # Load data
   df = pd.read_csv('credit_data.csv')
   
   # 1. Temporal features
   trend = TemporalTrendFeatures()
   df = trend.delta(df, 'balance', time_col='date', group_cols=['customer_id'])
   df = trend.momentum(df, 'balance', time_col='date', group_cols=['customer_id'], window=3)
   
   # 2. Categorical encoding
   encoder = CategoricalEncoder()
   df = encoder.freq_encoding(df, 'account_type')
   df = encoder.target_mean_encoding(df, 'region', 'target', smoothing=5.0)
   
   # 3. Validate features
   validator = FeatureValidator(warning_thresholds={'missing_rate': 0.05})
   results = validator.validate_features(df)
   validation_df = validator.to_dataframe()
   
   # 4. Manage dependencies
   graph = DependencyGraph()
   graph.add_feature('balance_delta', ['balance'])
   graph.add_feature('momentum', ['balance_delta'])
   execution_order = graph.topological_sort()
   
   # 5. Track in registry
   registry = FeatureRegistry()
   for feature in execution_order:
       registry.register(
           name=feature,
           source_columns=[...],
           operation='...',
           parameters={},
           window=None,
           missing_strategy='keep',
           dependencies=graph.get_dependencies(feature),
           engine='pandas',
           output_dtype='float64'
       )
   
   registry.export_json('audit/feature_registry.json')

Best Practices
--------------

1. **Temporal Features**: Always sort by time_col and group by entity ID
2. **Categorical Encoding**: Use smoothing in target mean encoding to avoid overfitting
3. **Validation**: Set appropriate thresholds based on business requirements
4. **Dependencies**: Keep dependency graphs acyclic and well-documented
5. **Registry**: Export registries after each run for audit trails

Performance Tips
----------------

For Pandas
~~~~~~~~~~

* Use vectorized operations when possible
* Sort once before multiple temporal operations
* Cache intermediate results for repeated use

For Large Datasets
~~~~~~~~~~~~~~~~~~

* Consider using Spark implementations for datasets >1M rows
* Enable caching in FeatureEngineeringConfig
* Batch similar operations together

See Also
--------

* :doc:`feature_engineering`: Core feature engineering capabilities
* :doc:`feature_selection`: Feature selection methods
* :doc:`/examples/enhanced_features`: Complete examples and notebooks
