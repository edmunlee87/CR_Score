Feature Selection Guide
=======================

A comprehensive guide to selecting the best features for your scorecard.

Why Feature Selection?
----------------------

Feature selection helps you:

- **Improve Model Performance**: Remove noisy or redundant features
- **Reduce Overfitting**: Fewer features = simpler, more generalizable models
- **Faster Scoring**: Less computation in production
- **Better Interpretability**: Easier to explain to stakeholders
- **Lower Maintenance**: Fewer features to monitor and maintain

When to Use Feature Selection
------------------------------

✅ **Use feature selection when:**

- You have many potential features (> 20)
- Some features have low Information Value (IV < 0.02)
- You need a simple, interpretable scorecard
- You want to reduce production complexity

❌ **Skip feature selection when:**

- You already have few features (< 10)
- All features have strong predictive power
- You need maximum accuracy regardless of complexity

Choosing a Selection Method
----------------------------

CR_Score provides 4 selection methods:

1. Forward Selection
~~~~~~~~~~~~~~~~~~~~

**How it works:**

- Starts with 0 features
- Adds one feature at a time
- Each iteration adds the feature that improves performance most
- Stops when no more improvement

**Pros:**

- Fast and efficient
- Good for many features
- Easy to understand

**Cons:**

- May miss feature interactions
- Greedy (not globally optimal)

**When to use:**

- You have many features (20+)
- You want quick results
- You prefer simplicity

**Example:**

.. code-block:: python

   from cr_score.features import ForwardSelector
   from sklearn.linear_model import LogisticRegression

   selector = ForwardSelector(
       estimator=LogisticRegression(random_state=42),
       max_features=10,           # Stop at 10 features
       min_improvement=0.001,     # Stop if improvement < 0.001
       scoring="roc_auc",         # Use AUC for scoring
       cv=5,                      # 5-fold cross-validation
       use_mlflow=True            # Track experiments
   )

   selector.fit(X_train, y_train)
   selected_features = selector.get_selected_features()
   print(f"Selected: {selected_features}")

2. Backward Elimination
~~~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

- Starts with ALL features
- Removes one feature at a time
- Each iteration removes the feature that hurts performance least
- Stops when you reach minimum features

**Pros:**

- Considers feature interactions
- Good starting point known

**Cons:**

- Slower than forward (evaluates all features first)
- May overfit initially

**When to use:**

- You have moderate features (10-30)
- You think most features are useful
- You want to consider interactions

**Example:**

.. code-block:: python

   from cr_score.features import BackwardSelector

   selector = BackwardSelector(
       estimator=LogisticRegression(random_state=42),
       min_features=5,            # Keep at least 5 features
       scoring="roc_auc",
       cv=5
   )

   selector.fit(X_train, y_train)

3. Stepwise Selection
~~~~~~~~~~~~~~~~~~~~~~

**How it works:**

- Combines forward and backward
- At each step, can ADD or REMOVE a feature
- Chooses the action that improves performance most
- Most flexible method

**Pros:**

- ✅ **RECOMMENDED** for most use cases
- Can correct mistakes from earlier steps
- Best balance of speed and accuracy
- Considers feature interactions

**Cons:**

- Slightly slower than forward-only

**When to use:**

- **Use this as your default choice!**
- Works well for any number of features
- Best overall performance

**Example:**

.. code-block:: python

   from cr_score.features import StepwiseSelector

   selector = StepwiseSelector(
       estimator=LogisticRegression(random_state=42),
       max_features=10,
       min_improvement=0.001,
       scoring="roc_auc",
       cv=5
   )

   selector.fit(X_train, y_train)
   selected = selector.get_selected_features()

4. Exhaustive Search
~~~~~~~~~~~~~~~~~~~~

**How it works:**

- Tries ALL possible feature combinations
- Finds the globally optimal subset
- Evaluates 2^N - 1 models for N features

**Pros:**

- Guaranteed to find the best combination
- Considers all interactions

**Cons:**

- ⚠️ **VERY SLOW!** Exponential complexity
- Only practical for small feature sets (N ≤ 15)

**When to use:**

- You have very few features (< 15)
- You need the absolute best combination
- You have time to wait

**Example:**

.. code-block:: python

   from cr_score.features import ExhaustiveSelector

   # ONLY use with small feature sets!
   selector = ExhaustiveSelector(
       estimator=LogisticRegression(random_state=42),
       min_features=3,
       max_features=5,            # Limit to 5 features max
       scoring="roc_auc",
       cv=5
   )

   # Only fit on top 10 features (from IV analysis)
   X_small = X_train[top_10_features]
   selector.fit(X_small, y_train)

Step-by-Step Tutorial
----------------------

Step 1: Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Load data
   df = pd.read_csv("applications.csv")

   # Split features and target
   X = df.drop(columns=["default"])
   y = df["default"]

   # Train/test split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )

   print(f"Training samples: {len(X_train)}")
   print(f"Features: {X_train.shape[1]}")

Step 2: Choose Your Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature selection is **model-agnostic** - works with ANY sklearn estimator:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.ensemble import RandomForestClassifier
   from xgboost import XGBClassifier

   # Option 1: Logistic Regression (recommended for scorecards)
   model = LogisticRegression(random_state=42, max_iter=1000)

   # Option 2: Random Forest
   model = RandomForestClassifier(n_estimators=100, random_state=42)

   # Option 3: XGBoost
   model = XGBClassifier(random_state=42)

   # Works with ANY sklearn-compatible model!

Step 3: Run Feature Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.features import StepwiseSelector

   # Create selector (use stepwise by default)
   selector = StepwiseSelector(
       estimator=model,
       max_features=10,           # Limit to 10 features
       min_improvement=0.001,     # Stop if improvement < 0.1%
       scoring="roc_auc",         # Optimize for AUC
       cv=5,                      # 5-fold cross-validation
       use_mlflow=True            # Track experiments
   )

   # Fit selector
   print("Running feature selection...")
   selector.fit(X_train, y_train)

   # Get results
   selected_features = selector.get_selected_features()
   best_score = selector.best_score_

   print(f"Selected {len(selected_features)} features:")
   for i, feat in enumerate(selected_features, 1):
       print(f"  {i}. {feat}")
   print(f"Best CV score: {best_score:.4f}")

Step 4: Transform Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transform to selected features only
   X_train_selected = selector.transform(X_train)
   X_test_selected = selector.transform(X_test)

   print(f"Original features: {X_train.shape[1]}")
   print(f"Selected features: {X_train_selected.shape[1]}")

Step 5: Train Final Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import roc_auc_score

   # Train model with selected features
   final_model = LogisticRegression(random_state=42)
   final_model.fit(X_train_selected, y_train)

   # Evaluate
   y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
   auc = roc_auc_score(y_test, y_pred_proba)

   print(f"Final AUC: {auc:.4f}")

MLflow Experiment Tracking
---------------------------

Track all experiments automatically with MLflow:

Setup MLflow
~~~~~~~~~~~~

.. code-block:: python

   import mlflow

   # Set experiment name
   mlflow.set_experiment("Scorecard_Feature_Selection")

   # Feature selection with MLflow tracking
   selector = StepwiseSelector(
       estimator=LogisticRegression(random_state=42),
       max_features=10,
       use_mlflow=True,           # Enable tracking
       mlflow_experiment_name="Scorecard_Feature_Selection"
   )

   selector.fit(X_train, y_train)

View Results
~~~~~~~~~~~~

.. code-block:: bash

   # Start MLflow UI
   mlflow ui

   # Open in browser: http://localhost:5000

MLflow tracks:

- Features tested at each iteration
- CV scores (mean and std)
- Model type used
- Number of features selected
- Best score achieved

Integration with Scorecard Pipeline
------------------------------------

Use feature selection in the complete pipeline:

.. code-block:: python

   from cr_score import ScorecardPipeline

   # Build scorecard with automatic feature selection
   pipeline = ScorecardPipeline(
       max_n_bins=5,
       feature_selection="stepwise",  # Enable feature selection
       max_features=10,               # Limit to 10 features
       pdo=20,
       base_score=600
   )

   pipeline.fit(df_train, target_col="default")

   # See which features were selected
   summary = pipeline.get_summary()
   print(f"Selected {summary['n_features']} features:")
   print(summary['selected_features'])

   # Evaluate
   metrics = pipeline.evaluate(df_test, target_col="default")
   print(f"AUC: {metrics['auc']:.3f}")

Best Practices
--------------

1. Start with IV Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Before feature selection, check Information Value:

.. code-block:: python

   from cr_score.encoding import WoEEncoder

   encoder = WoEEncoder()
   encoder.fit(df_binned, feature_cols=features, target_col="default")

   iv_summary = encoder.get_iv_summary()
   print(iv_summary.sort_values("iv", ascending=False))

   # Remove very weak features (IV < 0.02) before selection
   strong_features = iv_summary[iv_summary["iv"] >= 0.02]["feature"].tolist()

2. Use Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Always use CV to avoid overfitting:

.. code-block:: python

   selector = StepwiseSelector(
       estimator=model,
       cv=5,  # Use 5-fold CV (minimum)
       # cv=10 for smaller datasets
   )

3. Set Stopping Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~

Prevent selecting too many features:

.. code-block:: python

   selector = StepwiseSelector(
       max_features=15,           # Hard limit
       min_improvement=0.001,     # Stop if improvement < 0.1%
   )

4. Compare Multiple Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try different methods and compare:

.. code-block:: python

   methods = {
       "forward": ForwardSelector(estimator=model, max_features=10),
       "backward": BackwardSelector(estimator=model, min_features=3),
       "stepwise": StepwiseSelector(estimator=model, max_features=10),
   }

   results = {}
   for name, selector in methods.items():
       selector.fit(X_train, y_train)
       results[name] = {
           "n_features": len(selector.get_selected_features()),
           "score": selector.best_score_,
           "features": selector.get_selected_features()
       }

   # Compare results
   for name, result in results.items():
       print(f"{name}: {result['n_features']} features, score={result['score']:.4f}")

5. Validate on Hold-Out Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always validate final model on unseen data:

.. code-block:: python

   # Select features
   selector.fit(X_train, y_train)
   X_train_selected = selector.transform(X_train)
   X_test_selected = selector.transform(X_test)

   # Train final model
   final_model.fit(X_train_selected, y_train)

   # Validate on test set
   test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_selected)[:, 1])
   print(f"Test AUC: {test_auc:.4f}")

Common Issues and Solutions
----------------------------

Issue 1: Selection Takes Too Long
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Reduce feature space first

.. code-block:: python

   # Pre-filter by IV
   top_30_features = iv_summary.head(30)["feature"].tolist()
   X_train_filtered = X_train[top_30_features]

   # Then run selection
   selector.fit(X_train_filtered, y_train)

Issue 2: Selected Features Change Between Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Set random seed

.. code-block:: python

   selector = StepwiseSelector(
       estimator=LogisticRegression(random_state=42),  # Set seed
       cv=5
   )

Issue 3: Different Methods Give Different Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** This is normal! Compare on validation set

.. code-block:: python

   # Test each selection on validation set
   for name, selector in selectors.items():
       X_val_selected = selector.transform(X_val)
       val_score = roc_auc_score(y_val, model.predict_proba(X_val_selected)[:, 1])
       print(f"{name}: {val_score:.4f}")

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.features import StepwiseSelector
   from cr_score.encoding import WoEEncoder
   from sklearn.model_selection import train_test_split
   import pandas as pd

   # Load data
   df = pd.read_csv("applications.csv")

   # Split
   train, test = train_test_split(df, test_size=0.3, random_state=42)

   # Option 1: Quick approach with pipeline
   pipeline = ScorecardPipeline(
       feature_selection="stepwise",
       max_features=10,
       max_n_bins=5,
       pdo=20,
       base_score=600
   )

   pipeline.fit(train, target_col="default")
   scores = pipeline.predict(test)

   # See results
   summary = pipeline.get_summary()
   print(f"Selected features: {summary['selected_features']}")

   metrics = pipeline.evaluate(test, target_col="default")
   print(f"AUC: {metrics['auc']:.3f}")
   print(f"Gini: {metrics['gini']:.3f}")

Next Steps
----------

- :doc:`/api/features` - Complete API reference
- :doc:`/examples/feature_selection_mlflow` - MLflow tracking example
- :doc:`/guides/visualization` - Visualize feature importance
- :doc:`/api/pipeline` - Use in complete pipeline
