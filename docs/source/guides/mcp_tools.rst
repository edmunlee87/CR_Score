MCP Tools Guide
===============

Complete guide to using MCP (Model Context Protocol) tools for AI agent integration.

What are MCP Tools?
-------------------

MCP Tools are standardized interfaces that allow AI agents to interact with CR_Score programmatically.

**Think of them as:**

- Function calls that AI agents can execute
- APIs specifically designed for AI interaction
- Standardized input/output formats (JSON)
- Self-contained operations with clear purposes

**Why MCP Tools?**

- ✅ **Agent-Ready**: AI agents can call them directly
- ✅ **Standardized**: Consistent interface across all tools
- ✅ **Self-Documenting**: Complete schema definitions included
- ✅ **Error Handling**: Graceful error responses
- ✅ **JSON Responses**: Easy to parse and use

Available Tools
---------------

CR_Score provides 4 MCP tools:

1. **score_predict_tool**
   - Predict credit scores for new applications
   - Input: Data file + trained model
   - Output: Scores and statistics

2. **model_evaluate_tool**
   - Evaluate model performance on test data
   - Input: Test data + trained model
   - Output: Performance metrics

3. **feature_select_tool**
   - Automatically select best features
   - Input: Training data + target
   - Output: Selected features and scores

4. **binning_analyze_tool**
   - Analyze optimal binning for a feature
   - Input: Data + feature + target
   - Output: Binning table and IV

Tool 1: score_predict_tool
---------------------------

**Purpose:** Score new loan applications

Use Cases
~~~~~~~~~

- Score daily batch of new applications
- Re-score existing portfolio
- Generate predictions for testing

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import score_predict_tool

   result = score_predict_tool(
       data_path="new_applications.csv",
       model_path="models/scorecard_v1.pkl",
       output_path="predictions.csv"  # Optional
   )

   print(f"Scored {result['n_records']} applications")
   print(f"Mean score: {result['score_statistics']['mean']:.0f}")
   print(f"Score range: {result['score_statistics']['min']:.0f} - {result['score_statistics']['max']:.0f}")

Input Parameters
~~~~~~~~~~~~~~~~

**data_path** (required)
   Path to input data file containing features

   - Format: CSV or Parquet
   - Must have same features as training data
   - Can have additional columns (will be ignored)

**model_path** (required)
   Path to trained pipeline model file

   - Format: Pickle file (.pkl)
   - Created with ``pickle.dump(pipeline, f)``

**output_path** (optional)
   Where to save predictions

   - Format: CSV
   - Includes original data + scores + probabilities

Output Format
~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "success",
       "n_records": 1000,
       "score_statistics": {
           "mean": 650.5,
           "median": 648.0,
           "min": 420.0,
           "max": 820.0,
           "std": 85.3
       },
       "probability_statistics": {
           "mean": 0.042,
           "median": 0.038
       },
       "output_path": "predictions.csv"
   }

Error Handling
~~~~~~~~~~~~~~

If something goes wrong:

.. code-block:: python

   {
       "status": "error",
       "error": "File not found: new_applications.csv"
   }

Always check ``status`` first!

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import score_predict_tool
   import pandas as pd

   # Score new applications
   result = score_predict_tool(
       data_path="data/new_apps_jan2024.csv",
       model_path="models/retail_scorecard_v2.pkl",
       output_path="results/scores_jan2024.csv"
   )

   # Check if successful
   if result["status"] == "success":
       print(f"✓ Successfully scored {result['n_records']} applications")

       # Show statistics
       stats = result["score_statistics"]
       print(f"Average score: {stats['mean']:.0f}")
       print(f"Range: {stats['min']:.0f} - {stats['max']:.0f}")

       # Load scored data
       scored_df = pd.read_csv(result["output_path"])
       print(f"Saved to: {result['output_path']}")

       # Make decisions
       high_risk = scored_df[scored_df["credit_score"] < 580]
       print(f"High risk applications: {len(high_risk)}")

   else:
       print(f"✗ Error: {result['error']}")

Tool 2: model_evaluate_tool
----------------------------

**Purpose:** Evaluate model performance on test data

Use Cases
~~~~~~~~~

- Validate model on new data
- Monitor model performance over time
- Compare different models
- Generate performance reports

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import model_evaluate_tool

   result = model_evaluate_tool(
       data_path="validation_data.csv",
       model_path="models/scorecard_v1.pkl",
       target_col="default"
   )

   print(f"AUC: {result['metrics']['auc']:.3f}")
   print(f"Gini: {result['metrics']['gini']:.3f}")
   print(f"KS: {result['metrics']['ks']:.3f}")
   print(f"Model is {result['interpretation']['auc']}")

Input Parameters
~~~~~~~~~~~~~~~~

**data_path** (required)
   Path to test data with ground truth

**model_path** (required)
   Path to trained model

**target_col** (optional, default="default")
   Name of target column in data

Output Format
~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "success",
       "metrics": {
           "auc": 0.825,
           "gini": 0.650,
           "ks": 0.447,
           "accuracy": 0.892,
           "precision": 0.654,
           "recall": 0.423,
           "f1_score": 0.515
       },
       "interpretation": {
           "auc": "Excellent",
           "gini": "Discrimination power: 65.00%",
           "ks": "Max separation: 44.70%"
       },
       "n_records": 5000
   }

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

**AUC (Area Under ROC Curve):**

- ≥ 0.8: Excellent
- ≥ 0.7: Good
- ≥ 0.6: Fair
- < 0.6: Poor

**Gini Coefficient:**

- 2 * AUC - 1
- Measures discrimination power
- 0 = random, 1 = perfect

**KS Statistic:**

- ≥ 0.4: Excellent
- ≥ 0.3: Good
- ≥ 0.2: Fair
- < 0.2: Poor

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import model_evaluate_tool
   import json

   # Evaluate on multiple datasets
   datasets = {
       "Validation": "data/validation_2024q1.csv",
       "Out-of-Time": "data/oot_2024q2.csv",
       "Out-of-Sample": "data/oos_region2.csv"
   }

   results = {}

   for name, data_path in datasets.items():
       result = model_evaluate_tool(
           data_path=data_path,
           model_path="models/production_model.pkl",
           target_col="default"
       )

       if result["status"] == "success":
           results[name] = result["metrics"]
           print(f"{name:20s} AUC: {result['metrics']['auc']:.3f}")
       else:
           print(f"{name:20s} Error: {result['error']}")

   # Save results
   with open("evaluation_summary.json", "w") as f:
       json.dump(results, f, indent=2)

Tool 3: feature_select_tool
----------------------------

**Purpose:** Automatically select best features for scorecard

Use Cases
~~~~~~~~~

- Reduce feature space from many candidates
- Find optimal feature subset
- Compare different selection methods
- Automate feature engineering

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import feature_select_tool

   result = feature_select_tool(
       data_path="training_data.csv",
       target_col="default",
       method="stepwise",
       max_features=10,
       output_path="selected_features.csv"  # Optional
   )

   print(f"Selected {result['n_features_selected']} features from {result['n_features_total']}")
   print(f"Features: {result['selected_features']}")
   print(f"Best score: {result['best_score']:.3f}")

Input Parameters
~~~~~~~~~~~~~~~~

**data_path** (required)
   Path to training data

**target_col** (required)
   Name of target column

**method** (optional, default="stepwise")
   Selection method: "forward", "backward", or "stepwise"

**max_features** (optional, default=10)
   Maximum number of features to select

**output_path** (optional)
   Where to save selected feature names

Output Format
~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "success",
       "method": "stepwise",
       "n_features_total": 45,
       "n_features_selected": 8,
       "selected_features": [
           "payment_history_score",
           "debt_to_income_ratio",
           "credit_utilization",
           "age",
           "employment_length",
           "number_of_accounts",
           "recent_inquiries",
           "income"
       ],
       "best_score": 0.847,
       "output_path": "selected_features.csv"
   }

Choosing a Method
~~~~~~~~~~~~~~~~~

**Forward Selection:**

- Fast
- Good for many features
- Greedy approach

**Backward Elimination:**

- Starts with all features
- Good for moderate feature count
- Considers interactions

**Stepwise Selection:** (RECOMMENDED)

- Best balance
- Can add or remove
- Most flexible

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import feature_select_tool
   import pandas as pd

   # Try different methods
   methods = ["forward", "backward", "stepwise"]
   results = {}

   for method in methods:
       result = feature_select_tool(
           data_path="data/train.csv",
           target_col="default",
           method=method,
           max_features=10
       )

       if result["status"] == "success":
           results[method] = result
           print(f"{method:12s}: {result['n_features_selected']} features, "
                 f"score={result['best_score']:.4f}")

   # Use best method
   best_method = max(results.items(), key=lambda x: x[1]["best_score"])
   print(f"\nBest method: {best_method[0]}")
   print(f"Selected features: {best_method[1]['selected_features']}")

Tool 4: binning_analyze_tool
-----------------------------

**Purpose:** Analyze optimal binning for a feature

Use Cases
~~~~~~~~~

- Determine optimal bins for a feature
- Calculate Information Value
- Validate binning strategy
- Explore feature-target relationship

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import binning_analyze_tool

   result = binning_analyze_tool(
       data_path="training_data.csv",
       feature_col="age",
       target_col="default",
       max_bins=5,
       output_path="age_binning.csv"  # Optional
   )

   print(f"Feature: {result['feature']}")
   print(f"Number of bins: {result['n_bins']}")
   print(f"IV: {result['iv']:.3f} ({result['iv_strength']})")

Input Parameters
~~~~~~~~~~~~~~~~

**data_path** (required)
   Path to data

**feature_col** (required)
   Feature to analyze

**target_col** (required)
   Target column name

**max_bins** (optional, default=10)
   Maximum bins to create

**output_path** (optional)
   Where to save binning table

Output Format
~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "success",
       "feature": "age",
       "n_bins": 5,
       "iv": 0.245,
       "iv_strength": "Strong",
       "binning_table": [
           {
               "bin": "[18, 25)",
               "count": 1250,
               "event_rate": 0.082,
               "woe": 0.423,
               "iv_contribution": 0.045
           },
           # ... more bins ...
       ],
       "output_path": "age_binning.csv"
   }

IV Strength Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **< 0.02**: Weak (not predictive)
- **0.02 - 0.1**: Medium
- **0.1 - 0.3**: Strong
- **0.3 - 0.5**: Very Strong
- **≥ 0.5**: Suspicious (check for data leakage!)

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.tools import binning_analyze_tool
   import pandas as pd

   # Analyze all numeric features
   numeric_features = ["age", "income", "debt_ratio", "credit_score"]

   results = []

   for feature in numeric_features:
       result = binning_analyze_tool(
           data_path="data/train.csv",
           feature_col=feature,
           target_col="default",
           max_bins=5
       )

       if result["status"] == "success":
           results.append({
               "feature": feature,
               "iv": result["iv"],
               "strength": result["iv_strength"],
               "n_bins": result["n_bins"]
           })

           print(f"{feature:20s}: IV={result['iv']:.3f} ({result['iv_strength']})")

   # Create summary
   summary_df = pd.DataFrame(results)
   summary_df = summary_df.sort_values("iv", ascending=False)
   print("\nTop Features by IV:")
   print(summary_df)

AI Agent Integration
--------------------

Complete Agent Workflow
~~~~~~~~~~~~~~~~~~~~~~~

Here's how an AI agent might use all tools together:

.. code-block:: python

   def ai_agent_scorecard_workflow():
       """Complete scorecard development by AI agent"""

       print("Step 1: Select best features...")
       feature_result = feature_select_tool(
           data_path="train.csv",
           target_col="default",
           method="stepwise",
           max_features=10
       )

       if feature_result["status"] != "success":
           return {"error": feature_result["error"]}

       selected_features = feature_result["selected_features"]
       print(f"Selected {len(selected_features)} features")

       print("\nStep 2: Analyze binning for each feature...")
       binning_results = []
       for feature in selected_features:
           result = binning_analyze_tool(
               data_path="train.csv",
               feature_col=feature,
               target_col="default",
               max_bins=5
           )

           if result["status"] == "success":
               binning_results.append({
                   "feature": feature,
                   "iv": result["iv"]
               })
               print(f"  {feature}: IV={result['iv']:.3f}")

       print("\nStep 3: Train model...")
       # (Model training would happen here)
       # Assume model is saved to scorecard.pkl

       print("\nStep 4: Evaluate model...")
       eval_result = model_evaluate_tool(
           data_path="validation.csv",
           model_path="scorecard.pkl",
           target_col="default"
       )

       if eval_result["status"] == "success":
           print(f"  AUC: {eval_result['metrics']['auc']:.3f}")
           print(f"  Gini: {eval_result['metrics']['gini']:.3f}")

       print("\nStep 5: Score new applications...")
       score_result = score_predict_tool(
           data_path="new_applications.csv",
           model_path="scorecard.pkl",
           output_path="scores.csv"
       )

       if score_result["status"] == "success":
           print(f"  Scored {score_result['n_records']} applications")
           print(f"  Mean score: {score_result['score_statistics']['mean']:.0f}")

       return {
           "status": "success",
           "features_selected": len(selected_features),
           "model_auc": eval_result["metrics"]["auc"],
           "applications_scored": score_result["n_records"]
       }

   # Run workflow
   result = ai_agent_scorecard_workflow()

Error Handling Best Practices
------------------------------

Always Check Status
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = score_predict_tool(...)

   if result["status"] == "success":
       # Process results
       scores = result["score_statistics"]
   else:
       # Handle error
       print(f"Error: {result['error']}")
       # Maybe retry or alert user

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Try multiple methods, use first that works
   methods = ["stepwise", "forward", "backward"]

   for method in methods:
       result = feature_select_tool(
           data_path="train.csv",
           target_col="default",
           method=method
       )

       if result["status"] == "success":
           print(f"Success with {method}")
           break
   else:
       print("All methods failed")

Logging
~~~~~~~

.. code-block:: python

   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   result = model_evaluate_tool(...)

   if result["status"] == "success":
       logger.info(f"Model evaluation successful: AUC={result['metrics']['auc']:.3f}")
   else:
       logger.error(f"Model evaluation failed: {result['error']}")

Common Patterns
---------------

Pattern 1: Validation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_model_pipeline(model_path, test_data_path):
       """Validate model and generate report"""

       # Evaluate
       eval_result = model_evaluate_tool(
           data_path=test_data_path,
           model_path=model_path
       )

       if eval_result["status"] != "success":
           return {"status": "failed", "reason": "evaluation_failed"}

       # Check if meets threshold
       if eval_result["metrics"]["auc"] < 0.7:
           return {"status": "failed", "reason": "poor_performance"}

       # Score test set
       score_result = score_predict_tool(
           data_path=test_data_path,
           model_path=model_path,
           output_path="test_scores.csv"
       )

       return {
           "status": "passed",
           "auc": eval_result["metrics"]["auc"],
           "scores_generated": score_result["n_records"]
       }

Pattern 2: Feature Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def discover_best_features(data_path, target_col, top_n=20):
       """Find best features using multiple approaches"""

       # Get all features
       import pandas as pd
       df = pd.read_csv(data_path)
       all_features = [c for c in df.columns if c != target_col]

       # Analyze each feature
       feature_scores = []
       for feature in all_features:
           result = binning_analyze_tool(
               data_path=data_path,
               feature_col=feature,
               target_col=target_col
           )

           if result["status"] == "success":
               feature_scores.append({
                   "feature": feature,
                   "iv": result["iv"]
               })

       # Sort by IV
       feature_scores.sort(key=lambda x: x["iv"], reverse=True)

       # Return top N
       return feature_scores[:top_n]

Pattern 3: Batch Scoring Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_scoring_service(input_folder, model_path, output_folder):
       """Score all CSV files in input folder"""
       import os

       results = []

       for filename in os.listdir(input_folder):
           if filename.endswith(".csv"):
               input_path = os.path.join(input_folder, filename)
               output_path = os.path.join(output_folder, f"scored_{filename}")

               result = score_predict_tool(
                   data_path=input_path,
                   model_path=model_path,
                   output_path=output_path
               )

               results.append({
                   "file": filename,
                   "status": result["status"],
                   "records": result.get("n_records", 0)
               })

       return results

Next Steps
----------

- :doc:`/api/tools` - Complete API reference with schemas
- :doc:`/examples/complete_workflow` - See tools in action
- :doc:`/api/pipeline` - Build models to use with tools
