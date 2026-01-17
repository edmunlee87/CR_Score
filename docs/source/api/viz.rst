Visualization
=============

Interactive Plotly-based visualizations for scorecard analysis.

Binning Visualizer
------------------

.. autoclass:: cr_score.viz.bin_plots.BinningVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example:

.. code-block:: python

   from cr_score.viz import BinningVisualizer

   visualizer = BinningVisualizer()

   # Plot binning table
   fig = visualizer.plot_binning_table(
       binning_table,
       title="Age Binning Analysis"
   )
   fig.show()
   fig.write_html("age_binning.html")

   # Plot IV summary
   fig = visualizer.plot_iv_summary(iv_summary, top_n=15)
   fig.write_html("iv_summary.html")

   # Feature comparison
   fig = visualizer.plot_feature_comparison(
       df=df,
       feature_col="age_bin",
       target_col="default"
   )
   fig.show()

Score Visualizer
----------------

.. autoclass:: cr_score.viz.score_plots.ScoreVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example:

.. code-block:: python

   from cr_score.viz import ScoreVisualizer

   visualizer = ScoreVisualizer()

   # Score distribution
   fig = visualizer.plot_score_distribution(scores, y_test)
   fig.show()

   # ROC curve
   fig = visualizer.plot_roc_curve(y_test, probas, title="Model ROC")
   fig.write_html("roc_curve.html")

   # Calibration curve
   fig = visualizer.plot_calibration_curve(y_test, probas, n_bins=10)
   fig.show()

   # Confusion matrix
   predictions = (probas >= 0.5).astype(int)
   fig = visualizer.plot_confusion_matrix(y_test, predictions)
   fig.show()

   # Score bands
   fig = visualizer.plot_score_bands(scores, y_test, n_bands=10)
   fig.write_html("score_bands.html")

   # KS statistic
   fig = visualizer.plot_ks_statistic(y_test, probas)
   fig.show()

   # Comprehensive report
   fig = visualizer.create_model_report(y_test, probas, scores)
   fig.write_html("full_model_report.html")

Available Charts
----------------

Binning Charts
~~~~~~~~~~~~~~

- **plot_binning_table**: Comprehensive 4-panel binning analysis
  
  - Distribution (bar chart)
  - Event rate (line chart)
  - WoE values (bar chart)
  - IV contribution (bar chart)

- **plot_iv_summary**: Information Value ranking for all features
  
  - Color-coded by IV strength (weak/medium/strong/very strong)
  - Reference lines at IV thresholds

- **plot_feature_comparison**: Feature vs target comparison
  
  - Volume bars
  - Event rate line
  - Dual y-axis

Score Charts
~~~~~~~~~~~~

- **plot_score_distribution**: Score distribution by target class
- **plot_roc_curve**: ROC curve with AUC and KS statistic
- **plot_calibration_curve**: Calibration plot (predicted vs actual)
- **plot_confusion_matrix**: Confusion matrix heatmap
- **plot_score_bands**: Score band analysis with event rates
- **plot_ks_statistic**: KS curve showing maximum separation
- **create_model_report**: Comprehensive multi-panel report

Temporal Drift Visualization
----------------------------

Enhanced visualization capabilities for temporal drift analysis across multiple snapshots.

Bin-Level Temporal Drift
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Temporal bin drift with confidence bands
   fig = visualizer.plot_temporal_bin_drift(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       snapshot_values=["2024-01", "2024-06", "2024-12"],
       baseline_snapshot="2024-01",
       show_confidence_bands=True
   )
   
   # Delta vs baseline
   fig = visualizer.plot_bin_delta_vs_baseline(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       baseline_snapshot="2024-01"
   )

Distribution Shift (PSI)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PSI by feature across snapshots
   fig = visualizer.plot_psi_by_feature(
       df,
       feature_col="age",
       snapshot_col="month_end",
       baseline_snapshot="2024-01",
       n_bins=10
   )

Score-Level Temporal Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Score distribution over time
   fig = score_viz.plot_temporal_score_distribution(
       df,
       score_col="credit_score",
       snapshot_col="month_end",
       target_col="default",
       snapshot_values=["2024-01", "2024-06", "2024-12"]
   )
   
   # KS curve comparison
   fig = score_viz.plot_temporal_ks_comparison(
       df,
       score_col="credit_score",
       target_col="default",
       snapshot_col="month_end"
   )
   
   # Stability metrics dashboard
   fig = score_viz.plot_temporal_stability_metrics(
       df,
       score_col="credit_score",
       target_col="default",
       snapshot_col="month_end",
       approval_threshold=600
   )

Segmentation Support
~~~~~~~~~~~~~~~~~~~~

All temporal methods support segmentation:

.. code-block:: python

   fig = visualizer.plot_temporal_bin_drift(
       df,
       feature_col="age_bin",
       target_col="default",
       snapshot_col="month_end",
       segment_col="product_type",
       segment_values=["credit_card", "personal_loan"]
   )

Export with Metadata
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metadata = {
       "feature_name": "age_bin",
       "model_id": "v2.1",
       "snapshot_range": "2024-01 to 2024-12",
       "baseline_snapshot": "2024-01",
   }
   
   visualizer._export_figure_with_metadata(
       fig,
       path="reports/temporal_drift.html",
       format="html",
       metadata=metadata
   )

All Charts Features
~~~~~~~~~~~~~~~~~~~~

- Interactive Plotly visualizations
- Hover tooltips with detailed information
- Zoom and pan capabilities
- Save as HTML, PNG, or SVG
- Publication-ready styling
- Responsive design
- Temporal drift analysis across snapshots
- PSI visualization for distribution shift
- Score stability monitoring
