Configuration Guide
===================

Complete guide to configuring CR_Score for your scorecard projects.

Configuration Approaches
------------------------

CR_Score supports two ways to configure scorecards:

**1. Python API (Recommended for beginners)**

- Configure directly in code
- Easy to understand and modify
- Good for interactive development
- Best for Jupyter notebooks

**2. YAML Files (Recommended for production)**

- External configuration files
- Version controlled separately
- Environment-specific settings
- Best for deployment

Python API Configuration
------------------------

Basic Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline

   pipeline = ScorecardPipeline(
       max_n_bins=5,           # Maximum bins per feature
       min_iv=0.02,            # Minimum IV to include feature
       pdo=20,                 # Points to double odds
       base_score=600,         # Score at base odds
       base_odds=50.0,         # Odds at base score (2% default rate)
       target_bad_rate=0.05,   # Target bad rate for calibration
       calibrate=True,         # Enable calibration
       feature_selection="stepwise",  # Feature selection method
       max_features=10,        # Maximum features to select
       random_state=42         # Random seed for reproducibility
   )

   pipeline.fit(df_train, target_col="default")

Parameter Explanations
~~~~~~~~~~~~~~~~~~~~~~

Binning Parameters
^^^^^^^^^^^^^^^^^^

**max_n_bins** (int, default=5)
   Maximum number of bins to create per feature

   - Fewer bins = simpler model, easier to explain
   - More bins = more granular, potentially better performance
   - **Typical range**: 3-10
   - **Recommended**: 5

**min_iv** (float, default=0.02)
   Minimum Information Value to include a feature

   - Filters out weak predictors
   - IV < 0.02 = weak predictive power
   - **Recommended**: 0.02

.. code-block:: python

   # Conservative (only strong features)
   pipeline = ScorecardPipeline(min_iv=0.1)

   # Aggressive (include medium features)
   pipeline = ScorecardPipeline(min_iv=0.02)

Feature Selection Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**feature_selection** (str or None, default=None)
   Feature selection method

   - ``None``: No feature selection (use all features with IV > min_iv)
   - ``"forward"``: Forward selection
   - ``"backward"``: Backward elimination
   - ``"stepwise"``: Stepwise selection (recommended)

**max_features** (int or None, default=None)
   Maximum features to select

   - ``None``: No limit
   - Typical range: 5-15
   - **Recommended for simplicity**: 10

.. code-block:: python

   # Use all features (no selection)
   pipeline = ScorecardPipeline()

   # Select best 10 features using stepwise
   pipeline = ScorecardPipeline(
       feature_selection="stepwise",
       max_features=10
   )

Scaling Parameters
^^^^^^^^^^^^^^^^^^

**pdo** (int, default=20)
   Points to Double the Odds

   - Every PDO points, odds of default double
   - Smaller PDO = scores change faster
   - Larger PDO = scores change slower
   - **Typical values**: 20, 50, 100
   - **Industry standard**: 20

**base_score** (int, default=600)
   Score at base odds

   - Reference point for scoring scale
   - **FICO-style**: 600
   - **Custom**: Any value (e.g., 500, 700)

**base_odds** (float, default=50.0)
   Odds at base score

   - Odds = P(good) / P(bad) = (1-p) / p
   - base_odds = 50 means 2% default rate at base_score
   - base_odds = 19 means 5% default rate
   - base_odds = 99 means 1% default rate

.. code-block:: python

   # FICO-style scoring (common)
   pipeline = ScorecardPipeline(
       pdo=20,
       base_score=600,
       base_odds=50.0  # 2% default at 600
   )

   # More granular scale
   pipeline = ScorecardPipeline(
       pdo=50,          # Scores change slower
       base_score=700,
       base_odds=99.0   # 1% default at 700
   )

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^

**calibrate** (bool, default=True)
   Whether to calibrate intercept to target bad rate

   - ``True``: Adjust intercept to match target_bad_rate
   - ``False``: Use model's natural intercept

**target_bad_rate** (float or None, default=None)
   Target bad rate for calibration

   - ``None``: Use observed bad rate in training data
   - Float (0-1): Calibrate to specific rate
   - **Example**: 0.05 for 5% target

.. code-block:: python

   # Calibrate to observed rate
   pipeline = ScorecardPipeline(calibrate=True)

   # Calibrate to specific rate
   pipeline = ScorecardPipeline(
       calibrate=True,
       target_bad_rate=0.05  # 5% target
   )

   # No calibration
   pipeline = ScorecardPipeline(calibrate=False)

Other Parameters
^^^^^^^^^^^^^^^^

**random_state** (int, default=42)
   Random seed for reproducibility

   - Set to any integer for reproducible results
   - **Important for production**!

.. code-block:: python

   pipeline = ScorecardPipeline(random_state=42)

Configuration Presets
---------------------

Simple Scorecard (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quick, interpretable scorecards:

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=5,
       min_iv=0.02,
       pdo=20,
       base_score=600
   )

High-Performance Scorecard
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For maximum accuracy:

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=10,           # More granular bins
       min_iv=0.01,             # Include weaker features
       feature_selection="stepwise",
       max_features=15,         # More features
       pdo=20,
       base_score=600,
       calibrate=True
   )

Interpretable Scorecard
~~~~~~~~~~~~~~~~~~~~~~~

For easy explanation to business:

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=3,            # Very few bins
       min_iv=0.1,              # Only strong features
       feature_selection="stepwise",
       max_features=5,          # Very few features
       pdo=50,                  # Slower score changes
       base_score=600,
       calibrate=True
   )

YAML Configuration Files
-------------------------

YAML configuration provides more structure and is easier to version control.

Basic YAML Structure
~~~~~~~~~~~~~~~~~~~~~

See ``src/cr_score/templates/intermediate/config_template.yml``:

.. code-block:: yaml

   project:
     name: "retail_scorecard"
     owner: "risk_team"
     version: "1.0"

   data:
     sources:
       - path: "data/applications.parquet"
         format: "parquet"

   target:
     definition: "default_flag"
     horizon_months: 12

   binning:
     fine:
       method: "quantile"
       max_bins: 20
     coarse:
       monotonicity: true
       min_bin_size: 0.05

   model:
     type: "logistic"
     regularization: "l2"
     C: 1.0

   scaling:
     pdo: 20
     base_score: 600
     base_odds: 50.0

   calibration:
     enabled: true
     target_bad_rate: 0.05

Environment-Specific Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different configs for different environments:

**config_dev.yml** (development):

.. code-block:: yaml

   project:
     name: "scorecard_dev"

   data:
     sources:
       - path: "data/sample_1000.csv"  # Small sample

   model:
     quick_test: true

**config_prod.yml** (production):

.. code-block:: yaml

   project:
     name: "scorecard_prod"

   data:
     sources:
       - path: "s3://production/data/full_dataset.parquet"

   model:
     optimization: "full"

Best Practices
--------------

1. Always Set random_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reproducibility:

.. code-block:: python

   pipeline = ScorecardPipeline(random_state=42)

2. Start Simple, Then Optimize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Step 1: Baseline
   baseline = ScorecardPipeline()
   baseline.fit(train, target_col="default")
   baseline_auc = baseline.evaluate(test)["auc"]

   # Step 2: Add feature selection
   with_selection = ScorecardPipeline(
       feature_selection="stepwise",
       max_features=10
   )
   with_selection.fit(train, target_col="default")
   selection_auc = with_selection.evaluate(test)["auc"]

   # Compare
   print(f"Baseline: {baseline_auc:.3f}")
   print(f"With selection: {selection_auc:.3f}")

3. Calibrate to Business Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If business wants 3% approval rate
   target_bad_rate = 0.03

   pipeline = ScorecardPipeline(
       calibrate=True,
       target_bad_rate=target_bad_rate
   )

4. Document Your Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save configuration for reproducibility
   config = {
       "max_n_bins": 5,
       "feature_selection": "stepwise",
       "max_features": 10,
       "pdo": 20,
       "base_score": 600,
       "calibrate": True,
       "random_state": 42
   }

   import json
   with open("model_config.json", "w") as f:
       json.dump(config, f, indent=2)

5. Version Your Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add config_v1.yml
   git commit -m "Scorecard config v1.0"
   git tag config-v1.0

Common Scenarios
----------------

Scenario 1: Regulatory Compliance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need simple, explainable model:

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=3,            # Simple binning
       feature_selection="stepwise",
       max_features=5,          # Only 5 features
       pdo=20,
       base_score=600,
       calibrate=True,
       random_state=42
   )

Scenario 2: High-Volume Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need fast scoring:

.. code-block:: python

   pipeline = ScorecardPipeline(
       feature_selection="forward",  # Quick selection
       max_features=8,               # Fewer features = faster
       max_n_bins=4,                 # Simpler rules
       pdo=20,
       base_score=600,
       random_state=42
   )

Scenario 3: Maximizing Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accuracy is priority:

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=10,                # More granular
       feature_selection="stepwise", # Best method
       max_features=15,              # More features
       calibrate=True,
       target_bad_rate=None,         # Use observed rate
       random_state=42
   )

Troubleshooting
---------------

Issue: Model performs poorly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check:**

1. Increase max_features
2. Lower min_iv threshold
3. Try different feature_selection method

.. code-block:: python

   # Try with more features
   pipeline = ScorecardPipeline(
       max_features=15,  # Increased from 10
       min_iv=0.01       # Lowered from 0.02
   )

Issue: Model too complex
~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:**

1. Reduce max_features
2. Increase min_iv
3. Use forward selection (faster)

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_features=5,   # Reduced
       min_iv=0.1,       # Only strong features
       feature_selection="forward"
   )

Issue: Scores don't match expectations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Check:**

1. Verify base_score and base_odds
2. Check calibration settings
3. Validate PDO

.. code-block:: python

   # If scores too low, increase base_score
   pipeline = ScorecardPipeline(
       base_score=700,  # Instead of 600
       pdo=20
   )

Next Steps
----------

- :doc:`/api/pipeline` - Complete API reference
- :doc:`/examples/simple_scorecard` - See configuration in action
- :doc:`/guides/feature_selection` - Feature selection guide
- :doc:`/guides/quickstart` - Get started quickly
