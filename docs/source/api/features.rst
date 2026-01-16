Feature Selection
=================

Model-agnostic feature selection methods with MLflow integration.

Overview
--------

The feature selection module provides four selection methods that work with any
scikit-learn compatible estimator:

- **Forward Selection**: Greedy additive feature selection
- **Backward Elimination**: Greedy feature removal
- **Stepwise Selection**: Bidirectional (recommended)
- **Exhaustive Search**: Globally optimal for small feature sets

All methods support:

- Cross-validation based evaluation
- MLflow experiment tracking
- Any sklearn-compatible model
- Configurable stopping criteria

Base Selector
-------------

.. autoclass:: cr_score.features.selection.BaseFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Forward Selection
-----------------

.. autoclass:: cr_score.features.selection.ForwardSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example:

.. code-block:: python

   from cr_score.features import ForwardSelector
   from sklearn.linear_model import LogisticRegression

   selector = ForwardSelector(
       estimator=LogisticRegression(random_state=42),
       max_features=10,
       min_improvement=0.001,
       use_mlflow=True
   )

   selector.fit(X_train, y_train)
   selected_features = selector.get_selected_features()
   X_selected = selector.transform(X_test)

Backward Elimination
--------------------

.. autoclass:: cr_score.features.selection.BackwardSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example:

.. code-block:: python

   from cr_score.features import BackwardSelector
   from sklearn.ensemble import RandomForestClassifier

   selector = BackwardSelector(
       estimator=RandomForestClassifier(random_state=42),
       min_features=3,
       use_mlflow=True
   )

   selector.fit(X_train, y_train)

Stepwise Selection
------------------

.. autoclass:: cr_score.features.selection.StepwiseSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example:

.. code-block:: python

   from cr_score.features import StepwiseSelector

   selector = StepwiseSelector(
       estimator=LogisticRegression(random_state=42),
       max_features=10,
       use_mlflow=True
   )

   selector.fit(X_train, y_train)

Exhaustive Search
-----------------

.. autoclass:: cr_score.features.selection.ExhaustiveSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. warning::
   Exhaustive search evaluates 2^N - 1 models for N features.
   Only use with small feature sets (N <= 15).

Example:

.. code-block:: python

   from cr_score.features import ExhaustiveSelector

   # Only use with small feature sets!
   X_small = X_train[top_10_features]

   selector = ExhaustiveSelector(
       estimator=LogisticRegression(random_state=42),
       min_features=3,
       max_features=5,
       use_mlflow=True
   )

   selector.fit(X_small, y_train)
