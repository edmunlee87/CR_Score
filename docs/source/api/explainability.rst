Explainability
==============

Model explainability and interpretability for credit scorecards.

Overview
--------

The explainability module provides comprehensive tools for understanding model predictions:

- **SHAP Analysis**: Global and local explanations using SHAP values
- **Reason Codes**: Regulatory-compliant adverse action reasons (FCRA/ECOA)
- **Feature Importance**: Multi-method importance analysis

Key Features
------------

✅ **SHAP Integration**
   - TreeExplainer, LinearExplainer, KernelExplainer
   - Waterfall, force, and summary plots
   - Model-agnostic explanations

✅ **Regulatory Compliance**
   - FCRA-compliant reason codes
   - Adverse action notices
   - ECOA compliance

✅ **Multiple Methods**
   - Coefficient importance
   - Permutation importance
   - Drop-column importance
   - Correlation analysis

SHAP Explainer
--------------

.. automodule:: cr_score.explainability.shap_explainer
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import SHAPExplainer
   
   # Create explainer
   explainer = SHAPExplainer(model, model_type='tree')
   explainer.fit(X_train, sample_size=100)
   
   # Calculate SHAP values
   shap_values = explainer.explain(X_test)
   
   # Visualize
   explainer.plot_summary(X_test)
   explainer.plot_waterfall(X_test.iloc[0])
   
   # Get feature importance
   importance = explainer.get_feature_importance(X_test)

Reason Code Generator
---------------------

.. automodule:: cr_score.explainability.reason_codes
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import ReasonCodeGenerator
   
   # Create generator
   generator = ReasonCodeGenerator(model, feature_names)
   
   # Generate reasons for adverse decision
   reasons = generator.generate_reasons(
       x=application_features,
       score=580,
       threshold=620,
       num_reasons=4
   )
   
   # Generate complete adverse action notice
   notice = generator.generate_adverse_action_notice(
       application_id='APP123',
       applicant_name='John Doe',
       score=580,
       threshold=620,
       x=application_features,
       creditor_name='Your Bank'
   )

Feature Importance Analyzer
----------------------------

.. automodule:: cr_score.explainability.feature_importance
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import FeatureImportanceAnalyzer
   
   # Create analyzer
   analyzer = FeatureImportanceAnalyzer(model, feature_names)
   
   # Run multiple importance methods
   importance_results = analyzer.analyze(
       X_test,
       y_test,
       methods=['coefficient', 'permutation', 'correlation']
   )
   
   # Get consensus importance
   consensus = analyzer.get_consensus_importance(
       importance_results,
       method='rank_average'
   )
   
   # Visualize
   analyzer.plot_importance(consensus, top_n=20)
   
   # Export to Excel
   analyzer.export_importance(importance_results, 'importance.xlsx')

Best Practices
--------------

SHAP Analysis
~~~~~~~~~~~~~

1. **Sample Background Data**
   
   Use representative sample for background:
   
   .. code-block:: python
   
      explainer.fit(X_train, sample_size=100)  # Faster computation

2. **Choose Right Explainer**
   
   - Tree models → TreeExplainer (fastest)
   - Linear models → LinearExplainer
   - Complex models → KernelExplainer (slower but accurate)

3. **Validate Additivity**
   
   .. code-block:: python
   
      shap_values = explainer.explain(X_test, check_additivity=True)

Reason Codes
~~~~~~~~~~~~

1. **Customize Reason Code Map**
   
   .. code-block:: python
   
      custom_reasons = {
          'debt_to_income_ratio': ('RC01', 'High debt-to-income ratio'),
          'credit_utilization': ('RC02', 'High credit utilization'),
          # ... add more
      }
      
      generator = ReasonCodeGenerator(
          model,
          feature_names,
          reason_code_map=custom_reasons
      )

2. **Always Provide Top 4 Reasons**
   
   Regulatory best practice (FCRA):
   
   .. code-block:: python
   
      reasons = generator.generate_reasons(x, score, threshold, num_reasons=4)

3. **Document Compliance**
   
   Include regulatory references in notices

Feature Importance
~~~~~~~~~~~~~~~~~~

1. **Use Multiple Methods**
   
   Different methods capture different aspects:
   
   .. code-block:: python
   
      methods = ['coefficient', 'permutation', 'drop_column', 'correlation']
      results = analyzer.analyze(X, y, methods=methods)

2. **Get Consensus**
   
   Combine methods for robust importance:
   
   .. code-block:: python
   
      consensus = analyzer.get_consensus_importance(results, method='rank_average')

3. **Validate Against Domain Knowledge**
   
   Importance should align with business understanding

Integration with Pipeline
-------------------------

The explainability modules integrate seamlessly with :class:`~cr_score.pipeline.ScorecardPipeline`:

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.explainability import SHAPExplainer, ReasonCodeGenerator
   
   # Train pipeline
   pipeline = ScorecardPipeline()
   pipeline.fit(train_df, target_col='default')
   
   # Get WoE-encoded data (what model actually uses)
   X_woe = pipeline.woe_encoder_.transform(test_df)
   
   # SHAP explanations on WoE features
   explainer = SHAPExplainer(pipeline.model_)
   explainer.fit(X_woe)
   shap_values = explainer.explain(X_woe)
   
   # Reason codes on original features
   generator = ReasonCodeGenerator(
       pipeline.model_,
       feature_names=pipeline.selected_features_
   )
   
   # Score and generate reasons
   scores = pipeline.predict(test_df)
   for idx, row in test_df.iterrows():
       reasons = generator.generate_reasons(
           row[pipeline.selected_features_],
           scores[idx],
           threshold=620
       )

See Also
--------

- :doc:`/api/pipeline` - ScorecardPipeline integration
- :doc:`/api/features` - Feature selection
- :doc:`/api/model` - Model training
- :doc:`/guides/mcp_tools` - MCP tools for AI agents
