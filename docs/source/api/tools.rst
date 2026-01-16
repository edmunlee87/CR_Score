MCP Tools
=========

Model Context Protocol (MCP) tools for AI agent integration.

Overview
--------

CR_Score provides standardized MCP tools that AI agents can call to perform
scorecard operations. All tools return structured JSON responses with error handling.

score_predict_tool
------------------

.. autofunction:: cr_score.tools.mcp_tools.score_predict_tool

Example:

.. code-block:: python

   from cr_score.tools import score_predict_tool

   result = score_predict_tool(
       data_path="new_applications.csv",
       model_path="models/scorecard_v1.pkl",
       output_path="predictions.csv"
   )

   print(f"Scored {result['n_records']} records")
   print(f"Mean score: {result['score_statistics']['mean']:.0f}")
   print(f"Score range: {result['score_statistics']['min']:.0f} - {result['score_statistics']['max']:.0f}")

model_evaluate_tool
-------------------

.. autofunction:: cr_score.tools.mcp_tools.model_evaluate_tool

Example:

.. code-block:: python

   from cr_score.tools import model_evaluate_tool

   result = model_evaluate_tool(
       data_path="test_data.csv",
       model_path="models/scorecard_v1.pkl",
       target_col="default"
   )

   print(f"AUC: {result['metrics']['auc']:.3f}")
   print(f"Gini: {result['metrics']['gini']:.3f}")
   print(f"KS: {result['metrics']['ks']:.3f}")
   print(f"Interpretation: {result['interpretation']['auc']}")

feature_select_tool
-------------------

.. autofunction:: cr_score.tools.mcp_tools.feature_select_tool

Example:

.. code-block:: python

   from cr_score.tools import feature_select_tool

   result = feature_select_tool(
       data_path="train_data.csv",
       target_col="default",
       method="stepwise",
       max_features=10,
       output_path="selected_features.csv"
   )

   print(f"Selected {result['n_features_selected']} from {result['n_features_total']} features")
   print(f"Features: {result['selected_features']}")
   print(f"Best score: {result['best_score']:.3f}")

binning_analyze_tool
--------------------

.. autofunction:: cr_score.tools.mcp_tools.binning_analyze_tool

Example:

.. code-block:: python

   from cr_score.tools import binning_analyze_tool

   result = binning_analyze_tool(
       data_path="train_data.csv",
       feature_col="age",
       target_col="default",
       max_bins=5,
       output_path="age_binning.csv"
   )

   print(f"Feature: {result['feature']}")
   print(f"Number of bins: {result['n_bins']}")
   print(f"IV: {result['iv']:.3f} ({result['iv_strength']})")

MCP Tool Registry
-----------------

.. autodata:: cr_score.tools.mcp_tools.MCP_TOOLS
   :annotation:

The ``MCP_TOOLS`` dictionary contains complete MCP schema definitions for all tools:

.. code-block:: python

   {
       "score_predict": {
           "function": score_predict_tool,
           "name": "score_predict",
           "description": "Predict credit scores for new customers",
           "input_schema": { ... }
       },
       ...
   }

Response Format
---------------

All MCP tools return standardized JSON responses:

Success Response
~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "success",
       # ... tool-specific results ...
   }

Error Response
~~~~~~~~~~~~~~

.. code-block:: python

   {
       "status": "error",
       "error": "Error message describing what went wrong"
   }

AI Agent Integration
--------------------

These tools are designed for seamless AI agent integration:

1. **Standardized Interface**: All tools follow MCP specifications
2. **Structured Responses**: JSON format with consistent structure
3. **Error Handling**: Graceful error handling with descriptive messages
4. **Self-Documenting**: Complete schema definitions included
5. **File-Based I/O**: Work with file paths for easy data exchange

Example Agent Workflow
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # AI agent workflow for scorecard development

   # Step 1: Select best features
   features_result = feature_select_tool(
       data_path="train.csv",
       target_col="default",
       method="stepwise"
   )

   # Step 2: Analyze binning for selected features
   for feature in features_result['selected_features']:
       binning_result = binning_analyze_tool(
           data_path="train.csv",
           feature_col=feature,
           target_col="default"
       )
       print(f"{feature}: IV={binning_result['iv']:.3f}")

   # Step 3: Evaluate final model
   eval_result = model_evaluate_tool(
       data_path="test.csv",
       model_path="scorecard.pkl"
   )

   # Step 4: Score new applications
   score_result = score_predict_tool(
       data_path="new_apps.csv",
       model_path="scorecard.pkl",
       output_path="scores.csv"
   )
