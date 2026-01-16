Scorecard Pipeline
==================

Simplified interface for end-to-end scorecard development.

Overview
--------

The :class:`~cr_score.pipeline.ScorecardPipeline` class provides a simplified,
scikit-learn-like interface for building complete credit scorecards in just 3 lines of code.

It automatically handles:

- Optimal binning (via OptBinning)
- WoE transformation
- Feature selection (optional)
- Logistic regression modeling
- Intercept calibration
- PDO scaling

ScorecardPipeline
-----------------

.. autoclass:: cr_score.pipeline.ScorecardPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Basic Usage
-----------

.. code-block:: python

   from cr_score import ScorecardPipeline

   # Create pipeline
   pipeline = ScorecardPipeline(
       max_n_bins=5,          # Max 5 bins per feature
       min_iv=0.02,           # Minimum IV to include features
       pdo=20,                # Every 20 points, odds double
       base_score=600,        # Score 600 = base odds
       base_odds=50.0,        # 2% default rate at base score
       calibrate=True         # Calibrate intercept
   )

   # Fit on training data
   pipeline.fit(df_train, target_col="default")

   # Predict scores
   scores = pipeline.predict(df_test)
   probas = pipeline.predict_proba(df_test)

   # Evaluate
   metrics = pipeline.evaluate(df_test, target_col="default")
   print(f"AUC: {metrics['auc']:.3f}")

   # Get summary
   summary = pipeline.get_summary()
   print(f"Selected {summary['n_features']} features")

With Feature Selection
----------------------

.. code-block:: python

   pipeline = ScorecardPipeline(
       max_n_bins=5,
       feature_selection="stepwise",  # forward, backward, or stepwise
       max_features=10,               # Limit final features
       pdo=20,
       base_score=600
   )

   pipeline.fit(df_train, target_col="default")

   # See which features were selected
   summary = pipeline.get_summary()
   print(f"Selected features: {summary['selected_features']}")
   print(f"Selection method: {summary['feature_selection_method']}")

Export Scorecard
----------------

.. code-block:: python

   # Export scorecard specification
   scorecard_spec = pipeline.export_scorecard()

   # Save to JSON
   import json
   with open("scorecard_v1.json", "w") as f:
       json.dump(scorecard_spec, f, indent=2)

   # Save pipeline
   import pickle
   with open("pipeline_v1.pkl", "wb") as f:
       pickle.dump(pipeline, f)

Methods
-------

fit(df, target_col, sample_weight)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit the scorecard pipeline on training data.

**Parameters:**

- ``df`` (pd.DataFrame): Training data with features and target
- ``target_col`` (str): Name of target column
- ``sample_weight`` (pd.Series, optional): Sample weights for compressed data

**Returns:**

- ``self``: Returns self for method chaining

predict(df)
~~~~~~~~~~~

Predict credit scores for new data.

**Parameters:**

- ``df`` (pd.DataFrame): Feature data

**Returns:**

- ``np.ndarray``: Array of credit scores

predict_proba(df)
~~~~~~~~~~~~~~~~~

Predict default probabilities for new data.

**Parameters:**

- ``df`` (pd.DataFrame): Feature data

**Returns:**

- ``np.ndarray``: Array of default probabilities

evaluate(df, target_col)
~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate model performance on test data.

**Parameters:**

- ``df`` (pd.DataFrame): Test data with features and target
- ``target_col`` (str): Name of target column

**Returns:**

- ``dict``: Dictionary with performance metrics (AUC, Gini, KS, etc.)

get_summary()
~~~~~~~~~~~~~

Get pipeline summary with feature importance and configuration.

**Returns:**

- ``dict``: Dictionary with pipeline summary

export_scorecard()
~~~~~~~~~~~~~~~~~~

Export scorecard specification for production deployment.

**Returns:**

- ``dict``: Scorecard specification with binning, coefficients, and scaling parameters
