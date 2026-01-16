Model Explainability Guide
===========================

Complete guide for understanding and explaining scorecard predictions.

Overview
--------

Model explainability is critical for:

- **Regulatory Compliance**: FCRA/ECOA adverse action notices
- **Trust & Transparency**: Stakeholder confidence in model decisions
- **Model Debugging**: Understanding prediction patterns
- **Business Insights**: Feature importance and driver analysis

This guide covers three key explainability tools:

1. **SHAP Analysis**: Model-agnostic explanations
2. **Reason Codes**: Regulatory-compliant adverse action reasons
3. **Feature Importance**: Multi-method importance analysis

Quick Start
-----------

Basic Explainability Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.explainability import (
       SHAPExplainer,
       ReasonCodeGenerator,
       FeatureImportanceAnalyzer
   )
   
   # Train scorecard
   pipeline = ScorecardPipeline()
   pipeline.fit(train_df, target_col='default')
   
   # Get WoE-encoded features (what model actually sees)
   X_woe = pipeline.woe_encoder_.transform(test_df)
   
   # 1. SHAP Explanations
   explainer = SHAPExplainer(pipeline.model_, model_type='linear')
   explainer.fit(X_woe, sample_size=100)
   
   # Global importance
   importance = explainer.get_feature_importance(X_woe)
   print(importance)
   
   # Local explanation
   explainer.plot_waterfall(X_woe.iloc[0])
   
   # 2. Reason Codes (for declined applications)
   generator = ReasonCodeGenerator(
       pipeline.model_,
       feature_names=pipeline.selected_features_
   )
   
   reasons = generator.generate_reasons(
       x=test_df.iloc[0][pipeline.selected_features_],
       score=580,
       threshold=620,
       num_reasons=4
   )
   
   # 3. Feature Importance
   analyzer = FeatureImportanceAnalyzer(
       pipeline.model_,
       feature_names=pipeline.selected_features_
   )
   
   importance_results = analyzer.analyze(
       X_woe,
       test_df['default'],
       methods=['coefficient', 'permutation']
   )

SHAP Analysis
-------------

What is SHAP?
~~~~~~~~~~~~~

SHAP (SHapley Additive exPlanations) provides:

- **Additive**: Prediction = base_value + sum(SHAP values)
- **Consistent**: Higher SHAP value = more important
- **Local**: Explains individual predictions
- **Global**: Aggregates to feature importance

Setting Up SHAP
~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import SHAPExplainer
   
   # Choose appropriate explainer type
   explainer = SHAPExplainer(
       model=pipeline.model_,
       model_type='linear'  # or 'tree', 'kernel'
   )
   
   # Fit on background data
   explainer.fit(
       X_train_woe,
       sample_size=100  # Sample for efficiency
   )

Explainer Types
~~~~~~~~~~~~~~~

**LinearExplainer** (Fastest, for logistic regression):

.. code-block:: python

   explainer = SHAPExplainer(model, model_type='linear')

**TreeExplainer** (Fast, for tree-based models):

.. code-block:: python

   explainer = SHAPExplainer(model, model_type='tree')

**KernelExplainer** (Slowest, model-agnostic):

.. code-block:: python

   explainer = SHAPExplainer(model, model_type='kernel')

Global Explanations
~~~~~~~~~~~~~~~~~~~

Understanding overall feature importance:

.. code-block:: python

   # Calculate SHAP values
   shap_values = explainer.explain(X_test_woe)
   
   # Get feature importance
   importance = explainer.get_feature_importance(X_test_woe)
   
   print("Top 10 Most Important Features:")
   print(importance.head(10))
   
   # Visualize
   explainer.plot_summary(X_test_woe)

Local Explanations
~~~~~~~~~~~~~~~~~~

Explaining individual predictions:

.. code-block:: python

   # Explain single prediction
   sample = X_test_woe.iloc[0]
   
   # Waterfall plot (shows contribution of each feature)
   explainer.plot_waterfall(sample)
   
   # Force plot (interactive visualization)
   explainer.plot_force(sample)
   
   # Get numeric SHAP values
   shap_values = explainer.explain(sample.to_frame().T)
   print("SHAP values:", shap_values)

Dependence Plots
~~~~~~~~~~~~~~~~

Understanding feature effects:

.. code-block:: python

   # How does a feature affect predictions?
   explainer.plot_dependence(
       feature='debt_to_income_ratio_woe',
       X=X_test_woe
   )
   
   # With interaction effects
   explainer.plot_dependence(
       feature='debt_to_income_ratio_woe',
       X=X_test_woe,
       interaction_feature='credit_utilization_woe'
   )

Reason Codes (Adverse Action)
------------------------------

Regulatory Requirements
~~~~~~~~~~~~~~~~~~~~~~~

**FCRA (Fair Credit Reporting Act)**:
- Must provide specific reasons for adverse decisions
- Top 4 reasons recommended
- Must be understandable to consumers

**ECOA (Equal Credit Opportunity Act)**:
- Must notify applicants of adverse actions
- Include creditor name and contact
- Provide specific reasons

Setting Up Reason Code Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import ReasonCodeGenerator
   
   # Define reason code map (optional, auto-generated if not provided)
   reason_code_map = {
       'debt_to_income_ratio': ('RC01', 'High debt-to-income ratio'),
       'credit_utilization': ('RC02', 'High credit utilization'),
       'num_recent_inquiries': ('RC03', 'Too many recent credit inquiries'),
       'payment_history_score': ('RC04', 'Poor payment history'),
       'age_of_credit': ('RC05', 'Insufficient credit history length')
   }
   
   generator = ReasonCodeGenerator(
       model=pipeline.model_,
       feature_names=pipeline.selected_features_,
       reason_code_map=reason_code_map
   )

Generating Reason Codes
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For declined application
   application = test_df.iloc[0]
   score = pipeline.predict(application.to_frame().T)[0]
   
   # Generate top reasons for adverse decision
   reasons = generator.generate_reasons(
       x=application[pipeline.selected_features_],
       score=score,
       threshold=620,  # Approval threshold
       num_reasons=4   # Top 4 reasons
   )
   
   print("Adverse Action Reasons:")
   for i, reason in enumerate(reasons, 1):
       print(f"{i}. {reason['code']}: {reason['description']}")
       print(f"   Impact: {reason['impact']:.3f}")
       print(f"   Feature value: {reason['feature_value']:.3f}")

Complete Adverse Action Notice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate full FCRA-compliant notice
   notice = generator.generate_adverse_action_notice(
       application_id='APP123456',
       applicant_name='John Doe',
       score=score,
       threshold=620,
       x=application[pipeline.selected_features_],
       creditor_name='Your Bank',
       creditor_address='123 Bank St, City, State 12345',
       creditor_phone='1-800-555-BANK',
       num_reasons=4
   )
   
   print(notice)
   
   # Output:
   # ═══════════════════════════════════════════
   # ADVERSE ACTION NOTICE
   # ═══════════════════════════════════════════
   # 
   # Application ID: APP123456
   # Applicant Name: John Doe
   # Date: 2026-01-16
   # 
   # Your credit application has been declined.
   # Credit Score: 580 (Threshold: 620)
   # 
   # REASONS FOR THIS DECISION:
   # 
   # 1. RC01: High debt-to-income ratio
   # 2. RC02: High credit utilization
   # 3. RC03: Too many recent credit inquiries
   # 4. RC04: Poor payment history
   # ...

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate reasons for multiple applications
   declined_apps = test_df[test_df['score'] < 620]
   
   batch_reasons = generator.generate_batch_reasons(
       X=declined_apps[pipeline.selected_features_],
       scores=declined_apps['score'],
       threshold=620
   )
   
   # Export to CSV
   import pandas as pd
   
   results = []
   for app_id, reasons in zip(declined_apps.index, batch_reasons):
       for rank, reason in enumerate(reasons, 1):
           results.append({
               'application_id': app_id,
               'reason_rank': rank,
               'reason_code': reason['code'],
               'description': reason['description'],
               'impact': reason['impact']
           })
   
   pd.DataFrame(results).to_csv('adverse_action_reasons.csv', index=False)

Feature Importance
------------------

Why Multiple Methods?
~~~~~~~~~~~~~~~~~~~~~

Different methods capture different aspects:

- **Coefficient**: Model's learned weights
- **Permutation**: Shuffling impact on performance
- **Drop-Column**: Retraining without feature
- **Correlation**: Linear relationship with target

Using multiple methods provides robust, consensus importance.

Setting Up Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.explainability import FeatureImportanceAnalyzer
   
   analyzer = FeatureImportanceAnalyzer(
       model=pipeline.model_,
       feature_names=pipeline.selected_features_,
       random_state=42
   )

Running Multiple Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run all methods
   importance_results = analyzer.analyze(
       X=X_woe,
       y=y_test,
       methods=['coefficient', 'permutation', 'drop_column', 'correlation'],
       cv=5  # Cross-validation folds for permutation
   )
   
   # View results
   for method, importance_df in importance_results.items():
       print(f"\n{method.upper()} Importance:")
       print(importance_df.head(10))

Consensus Importance
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Combine methods using rank averaging
   consensus = analyzer.get_consensus_importance(
       importance_results,
       method='rank_average'  # or 'mean', 'median'
   )
   
   print("Consensus Feature Importance:")
   print(consensus.head(15))
   
   # Visualize
   analyzer.plot_importance(
       consensus,
       top_n=20,
       title='Consensus Feature Importance'
   )

Comparing Methods
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare all methods side-by-side
   analyzer.plot_importance_comparison(
       importance_results,
       top_n=15
   )

Exporting Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export to Excel (one sheet per method)
   analyzer.export_importance(
       importance_results,
       'feature_importance_analysis.xlsx'
   )

Integration with Scorecard Pipeline
------------------------------------

Complete Explainability Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.explainability import (
       SHAPExplainer,
       ReasonCodeGenerator,
       FeatureImportanceAnalyzer
   )
   
   # 1. Train scorecard
   pipeline = ScorecardPipeline()
   pipeline.fit(train_df, target_col='default')
   
   # 2. Get encoded features
   X_woe_train = pipeline.woe_encoder_.transform(train_df)
   X_woe_test = pipeline.woe_encoder_.transform(test_df)
   
   # 3. SHAP Analysis
   shap_explainer = SHAPExplainer(pipeline.model_, model_type='linear')
   shap_explainer.fit(X_woe_train, sample_size=100)
   
   # Global importance
   shap_importance = shap_explainer.get_feature_importance(X_woe_test)
   shap_explainer.plot_summary(X_woe_test)
   
   # Local explanations for top/bottom scores
   test_df['score'] = pipeline.predict(test_df)
   
   # Highest score
   best_idx = test_df['score'].idxmax()
   print(f"Highest score: {test_df.loc[best_idx, 'score']:.1f}")
   shap_explainer.plot_waterfall(X_woe_test.loc[best_idx])
   
   # Lowest score
   worst_idx = test_df['score'].idxmin()
   print(f"Lowest score: {test_df.loc[worst_idx, 'score']:.1f}")
   shap_explainer.plot_waterfall(X_woe_test.loc[worst_idx])
   
   # 4. Reason Codes for Declined Applications
   generator = ReasonCodeGenerator(
       pipeline.model_,
       feature_names=pipeline.selected_features_
   )
   
   declined = test_df[test_df['score'] < 620]
   print(f"\nAnalyzing {len(declined)} declined applications...")
   
   for idx in declined.index[:5]:  # First 5 examples
       app = test_df.loc[idx]
       reasons = generator.generate_reasons(
           x=app[pipeline.selected_features_],
           score=app['score'],
           threshold=620,
           num_reasons=4
       )
       
       print(f"\nApplication {idx} (Score: {app['score']:.1f}):")
       for rank, reason in enumerate(reasons, 1):
           print(f"  {rank}. {reason['description']}")
   
   # 5. Feature Importance Analysis
   analyzer = FeatureImportanceAnalyzer(
       pipeline.model_,
       feature_names=pipeline.selected_features_
   )
   
   importance_results = analyzer.analyze(
       X_woe_test,
       test_df['default'],
       methods=['coefficient', 'permutation', 'correlation']
   )
   
   consensus = analyzer.get_consensus_importance(
       importance_results,
       method='rank_average'
   )
   
   # Visualize
   analyzer.plot_importance_comparison(importance_results, top_n=15)
   
   # Export all results
   analyzer.export_importance(importance_results, 'importance_analysis.xlsx')
   
   print("\n✅ Explainability analysis complete!")

Best Practices
--------------

SHAP Analysis
~~~~~~~~~~~~~

1. **Choose Right Explainer**
   
   - Logistic regression → LinearExplainer
   - Random forest/XGBoost → TreeExplainer
   - Black box models → KernelExplainer

2. **Sample Background Data**
   
   .. code-block:: python
   
      # 100-1000 samples is usually sufficient
      explainer.fit(X_train, sample_size=100)

3. **Validate Additivity**
   
   .. code-block:: python
   
      # Ensure SHAP values sum to prediction
      shap_values = explainer.explain(X, check_additivity=True)

4. **Use WoE-Encoded Features**
   
   Explain what the model actually sees, not raw features

Reason Codes
~~~~~~~~~~~~

1. **Customize Reason Code Map**
   
   Make descriptions consumer-friendly:
   
   .. code-block:: python
   
      reason_code_map = {
          'debt_to_income_ratio': (
              'RC01',
              'Your debt payments compared to your income are too high'
          ),
          # Avoid technical jargon
      }

2. **Always Provide Top 4 Reasons**
   
   Regulatory best practice

3. **Store for Compliance**
   
   .. code-block:: python
   
      # Log all adverse actions
      reasons_log = {
          'application_id': app_id,
          'timestamp': datetime.now(),
          'score': score,
          'reasons': reasons
      }
      
      save_to_audit_trail(reasons_log)

4. **Test for Bias**
   
   Ensure reasons don't reveal protected characteristics

Feature Importance
~~~~~~~~~~~~~~~~~~

1. **Use Multiple Methods**
   
   Don't rely on single method

2. **Get Consensus**
   
   Rank averaging provides robust importance

3. **Validate Against Domain Knowledge**
   
   Importance should align with business understanding

4. **Monitor Over Time**
   
   .. code-block:: python
   
      # Track importance trends
      monthly_importance = {}
      for month, data in monthly_data.items():
          importance = analyzer.analyze(data['X'], data['y'])
          monthly_importance[month] = importance

Common Pitfalls
---------------

1. **Explaining Raw Features Instead of WoE**
   
   ❌ Wrong:
   
   .. code-block:: python
   
      explainer.fit(X_train)  # Raw features
   
   ✅ Correct:
   
   .. code-block:: python
   
      X_woe = pipeline.woe_encoder_.transform(X_train)
      explainer.fit(X_woe)  # WoE-encoded

2. **Not Sampling Background Data**
   
   ❌ Wrong (slow):
   
   .. code-block:: python
   
      explainer.fit(X_train)  # All 100K rows
   
   ✅ Correct (fast):
   
   .. code-block:: python
   
      explainer.fit(X_train, sample_size=100)

3. **Using Too Few Reason Codes**
   
   ❌ Wrong:
   
   .. code-block:: python
   
      reasons = generator.generate_reasons(..., num_reasons=2)
   
   ✅ Correct:
   
   .. code-block:: python
   
      reasons = generator.generate_reasons(..., num_reasons=4)

4. **Ignoring Correlation in Feature Importance**
   
   Highly correlated features may have split importance

Regulatory Compliance
---------------------

FCRA Compliance Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Adverse Action Notice**
   - Application/applicant identification
   - Clear decline statement
   - Specific reasons (top 4)
   - Creditor contact information
   - FCRA disclosure statement

✅ **Timely Notification**
   - Within 30 days of decision

✅ **Reason Code Quality**
   - Specific, not vague
   - Consumer-understandable
   - Actionable when possible

ECOA Compliance Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Notification Content**
   - Creditor identification
   - Applicant rights statement
   - Specific reasons for decline

✅ **No Discriminatory Factors**
   - Reasons must not reference protected characteristics

Documentation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Store for audit
   explainability_record = {
       'application_id': app_id,
       'timestamp': datetime.now(),
       'model_version': 'v2.0',
       'score': score,
       'threshold': threshold,
       'reasons': reasons,
       'shap_values': shap_values.tolist(),
       'feature_importance': importance.to_dict()
   }
   
   # Save to audit database
   save_to_compliance_log(explainability_record)

See Also
--------

- :doc:`/api/explainability` - Explainability API reference
- :doc:`/api/pipeline` - ScorecardPipeline integration
- :doc:`/guides/observability` - Production monitoring
- :doc:`/examples/complete_workflow` - Complete examples
