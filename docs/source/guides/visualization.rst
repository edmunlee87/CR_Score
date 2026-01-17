Visualization Guide
===================

Create professional, interactive visualizations for your scorecard analysis.

Why Visualize?
--------------

Visualizations help you:

- **Understand Your Data**: See patterns and relationships
- **Validate Binning**: Check if bins make business sense
- **Assess Model Performance**: Evaluate predictive power
- **Communicate Results**: Share insights with stakeholders
- **Identify Issues**: Spot problems early

What Can You Visualize?
------------------------

CR_Score provides two main visualizers:

**1. BinningVisualizer** - For binning analysis

- Bin distributions
- Event rates across bins
- WoE (Weight of Evidence) values
- IV (Information Value) contributions
- Feature comparisons

**2. ScoreVisualizer** - For model performance

- Score distributions
- ROC curves
- Calibration plots
- Confusion matrices
- Score band analysis
- KS statistics

All visualizations are:

- ✅ **Interactive** (zoom, pan, hover)
- ✅ **Plotly-based** (publication quality)
- ✅ **Exportable** (HTML, PNG, SVG)
- ✅ **Customizable** (titles, colors, sizes)

Getting Started
---------------

Basic Setup
~~~~~~~~~~~

.. code-block:: python

   from cr_score.viz import BinningVisualizer, ScoreVisualizer

   # Create visualizers
   bin_viz = BinningVisualizer()
   score_viz = ScoreVisualizer()

Binning Visualizations
-----------------------

1. Complete Binning Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a comprehensive 4-panel binning chart:

.. code-block:: python

   from cr_score.viz import BinningVisualizer
   from cr_score.encoding import WoEEncoder

   # Get binning table from WoE encoder
   encoder = WoEEncoder()
   encoder.fit(df_binned, feature_cols=["age_bin"], target_col="default")
   binning_table = encoder.get_woe_table("age_bin")

   # Create visualization
   visualizer = BinningVisualizer()
   fig = visualizer.plot_binning_table(
       binning_table,
       title="Age Binning Analysis"
   )

   # Display
   fig.show()

   # Or save to file
   fig.write_html("age_binning.html")

**What you get:**

- **Top Left**: Distribution (bar chart of sample counts)
- **Top Right**: Event rate (line chart showing default rate)
- **Bottom Left**: WoE values (bar chart)
- **Bottom Right**: IV contribution (bar chart)

2. Information Value Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare IV across all features:

.. code-block:: python

   # Get IV summary
   iv_summary = encoder.get_iv_summary()

   # Visualize top features
   fig = bin_viz.plot_iv_summary(
       iv_summary,
       top_n=15  # Show top 15 features
   )

   fig.show()

**Features:**

- Color-coded by IV strength:
  
  - Gray: Weak (IV < 0.02)
  - Yellow: Medium (0.02 ≤ IV < 0.1)
  - Orange: Strong (0.1 ≤ IV < 0.3)
  - Green: Very Strong (0.3 ≤ IV < 0.5)
  - Red: Suspicious (IV ≥ 0.5, check for data leakage!)

- Reference lines at IV thresholds
- Sorted by IV value

3. Feature Comparison
~~~~~~~~~~~~~~~~~~~~~

Compare feature values vs target:

.. code-block:: python

   fig = bin_viz.plot_feature_comparison(
       df=df,
       feature_col="age_bin",
       target_col="default"
   )

   fig.show()

**Shows:**

- Volume (bar chart)
- Event rate (line chart on secondary axis)
- Helps validate if binning makes business sense

Score Visualizations
--------------------

1. Score Distribution
~~~~~~~~~~~~~~~~~~~~~

See how scores are distributed for good vs bad customers:

.. code-block:: python

   from cr_score.viz import ScoreVisualizer

   # Predict scores
   scores = pipeline.predict(test_df)
   y_true = test_df["default"]

   # Visualize
   visualizer = ScoreVisualizer()
   fig = visualizer.plot_score_distribution(
       scores=scores,
       y_true=y_true,
       n_bins=20
   )

   fig.show()

**What to look for:**

- Good separation between good (green) and bad (red) customers
- Minimal overlap
- Bad customers should have lower scores

2. ROC Curve
~~~~~~~~~~~~

Evaluate model discrimination:

.. code-block:: python

   # Get probabilities
   probas = pipeline.predict_proba(test_df)

   # Plot ROC
   fig = score_viz.plot_roc_curve(
       y_true=y_test,
       y_pred_proba=probas,
       title="Scorecard ROC Curve"
   )

   fig.show()

**Includes:**

- ROC curve (blue line)
- Random baseline (gray diagonal)
- AUC score in legend
- KS statistic marked with red dot

**Interpretation:**

- AUC ≥ 0.8: Excellent
- AUC ≥ 0.7: Good
- AUC ≥ 0.6: Fair
- AUC < 0.6: Poor (model not useful)

3. Calibration Curve
~~~~~~~~~~~~~~~~~~~~~

Check if predicted probabilities match actual rates:

.. code-block:: python

   fig = score_viz.plot_calibration_curve(
       y_true=y_test,
       y_pred_proba=probas,
       n_bins=10
   )

   fig.show()

**What to look for:**

- Points should be close to diagonal line
- If above line: model is under-predicting
- If below line: model is over-predicting

4. Confusion Matrix
~~~~~~~~~~~~~~~~~~~~

See classification accuracy:

.. code-block:: python

   # Convert probabilities to predictions
   predictions = (probas >= 0.5).astype(int)

   # Plot confusion matrix
   fig = score_viz.plot_confusion_matrix(
       y_true=y_test,
       y_pred=predictions
   )

   fig.show()

**Shows:**

- True Negatives (TN): Correctly predicted good
- False Positives (FP): Predicted bad, actually good
- False Negatives (FN): Predicted good, actually bad
- True Positives (TP): Correctly predicted bad

5. Score Bands
~~~~~~~~~~~~~~

Analyze performance across score ranges:

.. code-block:: python

   fig = score_viz.plot_score_bands(
       scores=scores,
       y_true=y_test,
       n_bands=10
   )

   fig.show()

**Shows:**

- Volume per score band (bars)
- Default rate per band (line)
- Helps validate score interpretation

**What to look for:**

- Default rate should decrease as score increases
- Monotonic relationship
- Adequate volume in each band

6. KS Statistic
~~~~~~~~~~~~~~~

Visualize maximum separation:

.. code-block:: python

   fig = score_viz.plot_ks_statistic(
       y_true=y_test,
       y_pred_proba=probas
   )

   fig.show()

**Shows:**

- Cumulative % of bad customers (red)
- Cumulative % of good customers (green)
- Maximum separation (KS statistic) marked

**Interpretation:**

- KS ≥ 40: Excellent
- KS ≥ 30: Good
- KS ≥ 20: Fair
- KS < 20: Poor

7. Comprehensive Report
~~~~~~~~~~~~~~~~~~~~~~~

Create a multi-panel report with all key metrics:

.. code-block:: python

   fig = score_viz.create_model_report(
       y_true=y_test,
       y_pred_proba=probas,
       scores=scores  # Optional
   )

   fig.write_html("model_report.html")

**Includes:**

- ROC curve
- Calibration curve
- Score distribution (if scores provided)
- KS statistic curve

**Perfect for:** Sharing with stakeholders

Complete Workflow Example
--------------------------

Here's a complete example from data to visualizations:

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.viz import BinningVisualizer, ScoreVisualizer
   import pandas as pd
   from sklearn.model_selection import train_test_split

   # 1. Load and split data
   df = pd.read_csv("applications.csv")
   train, test = train_test_split(df, test_size=0.3, random_state=42)

   # 2. Build scorecard
   pipeline = ScorecardPipeline(
       max_n_bins=5,
       feature_selection="stepwise",
       max_features=10,
       pdo=20,
       base_score=600
   )

   pipeline.fit(train, target_col="default")

   # 3. Get predictions
   scores = pipeline.predict(test)
   probas = pipeline.predict_proba(test)
   y_test = test["default"]

   # 4. Create visualizations
   bin_viz = BinningVisualizer()
   score_viz = ScoreVisualizer()

   # Binning analysis
   summary = pipeline.get_summary()
   for feature in summary["selected_features"]:
       binning_table = pipeline.auto_binner_.get_woe_table(feature)
       fig = bin_viz.plot_binning_table(
           binning_table,
           title=f"{feature} Binning"
       )
       fig.write_html(f"binning_{feature}.html")

   # IV summary
   iv_summary = pipeline.auto_binner_.get_iv_summary()
   fig = bin_viz.plot_iv_summary(iv_summary)
   fig.write_html("iv_summary.html")

   # Model performance
   fig = score_viz.plot_roc_curve(y_test, probas)
   fig.write_html("roc_curve.html")

   fig = score_viz.plot_score_distribution(scores, y_test)
   fig.write_html("score_distribution.html")

   fig = score_viz.plot_score_bands(scores, y_test)
   fig.write_html("score_bands.html")

   # Comprehensive report
   fig = score_viz.create_model_report(y_test, probas, scores)
   fig.write_html("full_report.html")

   print("All visualizations saved!")

Customization Options
----------------------

Customize Appearance
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig = score_viz.plot_roc_curve(y_test, probas)

   # Update layout
   fig.update_layout(
       title="My Custom Title",
       title_font_size=20,
       width=800,
       height=600,
       template="plotly_dark"  # Dark theme
   )

   # Update traces
   fig.update_traces(
       line=dict(color="red", width=3)
   )

   fig.show()

Save in Different Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # HTML (interactive)
   fig.write_html("chart.html")

   # PNG (static image)
   fig.write_image("chart.png", width=1200, height=800)

   # SVG (vector graphics)
   fig.write_image("chart.svg")

   # PDF
   fig.write_image("chart.pdf")

Note: PNG/SVG/PDF require ``kaleido`` package:

.. code-block:: bash

   pip install kaleido

Best Practices
--------------

1. Always Visualize Before Deciding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't just look at numbers - visualize!

.. code-block:: python

   # Bad: Just checking IV number
   if iv > 0.1:
       use_feature = True

   # Good: Visualize and validate
   fig = bin_viz.plot_binning_table(binning_table)
   fig.show()
   # Then decide if binning makes business sense

2. Use Consistent Color Schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the same colors for good/bad across all charts:

- Green for good customers
- Red for bad customers

3. Add Context with Titles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fig = score_viz.plot_roc_curve(
       y_test, probas,
       title="Retail Scorecard - Validation Set (N=10,000)"
   )

4. Save Both HTML and PNG
~~~~~~~~~~~~~~~~~~~~~~~~~~

- HTML for interactive exploration
- PNG for presentations and documents

.. code-block:: python

   fig.write_html("chart_interactive.html")
   fig.write_image("chart_static.png", width=1200, height=800)

5. Create a Visualization Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine multiple charts in one HTML:

.. code-block:: python

   from plotly.subplots import make_subplots

   fig = make_subplots(
       rows=2, cols=2,
       subplot_titles=("ROC Curve", "Score Distribution",
                      "Calibration", "Score Bands")
   )

   # Add individual plots...
   # (Advanced - see Plotly documentation)

Common Issues and Solutions
----------------------------

Issue 1: Charts Not Showing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``fig.show()`` doesn't display

**Solution:** Make sure you're in an environment that supports Plotly:

.. code-block:: python

   # In Jupyter: Should work automatically
   fig.show()

   # In script: Opens in browser
   fig.show()

   # Alternative: Save to HTML and open
   fig.write_html("temp.html")
   import webbrowser
   webbrowser.open("temp.html")

Issue 2: Charts Look Cluttered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Too much data on one chart

**Solution:** Filter or aggregate:

.. code-block:: python

   # Show only top features
   iv_summary_top = iv_summary.head(10)
   fig = bin_viz.plot_iv_summary(iv_summary_top)

   # Reduce number of bins
   fig = score_viz.plot_score_bands(scores, y_test, n_bands=5)

Issue 3: Export Fails
~~~~~~~~~~~~~~~~~~~~~~

**Problem:** ``fig.write_image()`` fails

**Solution:** Install kaleido:

.. code-block:: bash

   pip install kaleido

Tips for Presenting to Stakeholders
------------------------------------

1. Start with the Big Picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Show overall performance first
   fig = score_viz.create_model_report(y_test, probas, scores)
   fig.write_html("executive_summary.html")

2. Then Show Feature Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Show IV summary to highlight important features
   fig = bin_viz.plot_iv_summary(iv_summary)

   # Then detailed binning for top features
   for feature in top_features:
       fig = bin_viz.plot_binning_table(...)

3. Use Score Bands for Business Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Score bands are easier to understand than probabilities
   fig = score_viz.plot_score_bands(scores, y_test, n_bands=10)

   # Add business labels
   # "Excellent", "Good", "Fair", "Poor", "Very Poor"

4. Include Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add annotations to explain what charts mean:

.. code-block:: python

   fig = score_viz.plot_roc_curve(y_test, probas)

   fig.add_annotation(
       text="AUC of 0.82 indicates excellent<br>discrimination between good and bad",
       xref="paper", yref="paper",
       x=0.5, y=0.95,
       showarrow=False,
       font=dict(size=12)
   )

Temporal Drift Visualization
----------------------------

Monitor how your features and scores change over time across multiple snapshots.

Why Temporal Visualization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temporal visualizations help you:

- **Detect Drift**: Identify when feature distributions shift
- **Monitor Stability**: Ensure model performance remains consistent
- **Validate Changes**: Compare current vs historical behavior
- **Compliance**: Track changes over time for regulatory reporting

Bin-Level Temporal Drift
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how bin behavior changes across snapshots:

.. code-block:: python

   from cr_score.viz import BinningVisualizer

   viz = BinningVisualizer()

   # Plot temporal drift with confidence bands
   fig = viz.plot_temporal_bin_drift(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       snapshot_values=["2024-01", "2024-06", "2024-12"],
       baseline_snapshot="2024-01",
       show_confidence_bands=True
   )
   fig.show()

**What you see:**

- **Top Panel**: Event rate per bin across snapshots (multi-line plot)
- **Bottom Panel**: Population % per bin across snapshots (stacked bars)
- **Confidence Bands**: Statistical uncertainty around event rates

Delta vs Baseline
~~~~~~~~~~~~~~~~~

See how bins change relative to a baseline period:

.. code-block:: python

   # Plot delta (change) vs baseline
   fig = viz.plot_bin_delta_vs_baseline(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       baseline_snapshot="2024-01"
   )
   fig.show()

**What you see:**

- **Left Panel**: Δ Event Rate vs baseline per bin
- **Right Panel**: Δ Population % vs baseline per bin
- **Zero Line**: Reference for no change

PSI Visualization
~~~~~~~~~~~~~~~~~

Track distribution shift using Population Stability Index:

.. code-block:: python

   # Plot PSI over time
   fig = viz.plot_psi_by_feature(
       df,
       feature_col="age",
       snapshot_col="month_end",
       baseline_snapshot="2024-01",
       n_bins=10
   )
   fig.show()

**PSI Interpretation:**

- **PSI < 0.10**: Low drift (acceptable)
- **PSI 0.10-0.25**: Medium drift (monitor closely)
- **PSI > 0.25**: High drift (action required)

Score Stability Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor score distribution and stability metrics:

.. code-block:: python

   from cr_score.viz import ScoreVisualizer

   score_viz = ScoreVisualizer()

   # Score distribution over time
   fig = score_viz.plot_temporal_score_distribution(
       df,
       score_col="credit_score",
       snapshot_col="month_end",
       target_col="default",
       snapshot_values=["2024-01", "2024-06", "2024-12"]
   )
   fig.show()

   # KS curve comparison
   fig = score_viz.plot_temporal_ks_comparison(
       df,
       score_col="credit_score",
       target_col="default",
       snapshot_col="month_end"
   )
   fig.show()

   # Stability metrics dashboard
   fig = score_viz.plot_temporal_stability_metrics(
       df,
       score_col="credit_score",
       target_col="default",
       snapshot_col="month_end",
       approval_threshold=600
   )
   fig.show()

**Stability Metrics Dashboard shows:**

- **Approval Rate**: % of applications approved over time
- **Bad Rate**: Default rate over time
- **Capture Rate (Top 10%)**: % of defaults captured in top decile
- **Capture Rate (Top 20%)**: % of defaults captured in top quintile

Segmentation Support
~~~~~~~~~~~~~~~~~~~~

Analyze temporal drift by segment:

.. code-block:: python

   fig = viz.plot_temporal_bin_drift(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       segment_col="product_type",
       segment_values=["credit_card", "personal_loan"]
   )
   fig.show()

Export with Metadata
~~~~~~~~~~~~~~~~~~~~

Export temporal visualizations with audit metadata:

.. code-block:: python

   metadata = {
       "feature_name": "age_bin",
       "model_id": "v2.1",
       "snapshot_range": "2024-01 to 2024-12",
       "baseline_snapshot": "2024-01",
       "segment": "consumer_portfolio"
   }

   viz._export_figure_with_metadata(
       fig,
       path="reports/temporal_drift_age_bin.html",
       format="html",
       metadata=metadata
   )

Metadata is embedded in the visualization for audit trails.

See Also
~~~~~~~~

- :doc:`/api/viz` - Complete API reference including temporal methods
- :doc:`/guides/enhanced_features` - Feature engineering for temporal features
- :doc:`/playbooks/09_temporal_visualization` - Interactive tutorial

Next Steps
----------

- :doc:`/api/viz` - Complete API reference
- :doc:`/guides/reporting` - Generate HTML reports
- :doc:`/guides/enhanced_features` - Advanced feature engineering
- :doc:`/examples/complete_workflow` - See visualizations in action
