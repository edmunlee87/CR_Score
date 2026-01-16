Reporting & Dashboards
=======================

Professional reporting and interactive dashboards for scorecards.

Overview
--------

The reporting module provides:

- **HTML Report Generation**: Comprehensive scorecard documentation
- **Observability Dashboards**: Real-time production monitoring
- **Interactive Visualizations**: Plotly-based charts
- **Export Formats**: HTML, PDF, Excel

HTML Report Generator
---------------------

.. automodule:: cr_score.reporting.html_report
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.reporting import HTMLReportGenerator
   
   # Train pipeline
   pipeline = ScorecardPipeline()
   pipeline.fit(train_df, target_col='default')
   
   # Generate report
   report_gen = HTMLReportGenerator()
   
   html_report = report_gen.generate(
       pipeline=pipeline,
       test_df=test_df,
       target_col='default',
       report_title="Credit Scorecard Report",
       author="Data Science Team",
       include_plots=True
   )
   
   # Save report
   with open('scorecard_report.html', 'w', encoding='utf-8') as f:
       f.write(html_report)

Observability Dashboard
-----------------------

.. automodule:: cr_score.reporting.observability_dashboard
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.reporting import ObservabilityDashboard
   from cr_score.monitoring import (
       PerformanceMonitor,
       DriftMonitor,
       MetricsCollector,
       AlertManager
   )
   
   # Initialize monitors (in production)
   perf_monitor = PerformanceMonitor(baseline_metrics={'auc': 0.85})
   drift_monitor = DriftMonitor(reference_data=X_train)
   metrics_collector = MetricsCollector()
   alert_manager = AlertManager()
   
   # ... production scoring and monitoring ...
   
   # Create dashboard
   dashboard = ObservabilityDashboard(
       title="Production Scorecard Monitoring Dashboard"
   )
   
   # Add performance section
   metrics_df = perf_monitor.get_metrics_summary()
   health = perf_monitor.check_health()
   dashboard.add_performance_section(metrics_df, health)
   
   # Add drift section
   drift_report = drift_monitor.detect_drift(X_production)
   dashboard.add_drift_section(drift_report)
   
   # Add prediction monitoring
   pred_stats = pred_monitor.get_prediction_summary()
   dashboard.add_prediction_section(pred_stats)
   
   # Add system metrics
   metrics = metrics_collector.get_metrics()
   dashboard.add_metrics_section(metrics)
   
   # Add active alerts
   active_alerts = alert_manager.get_active_alerts()
   alert_summary = alert_manager.get_alert_summary()
   dashboard.add_alerts_section(active_alerts, alert_summary)
   
   # Export dashboard
   dashboard.export('monitoring_dashboard.html')

Complete Reporting Workflow
----------------------------

Development Phase
~~~~~~~~~~~~~~~~~

Use HTMLReportGenerator for model development and validation:

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.reporting import HTMLReportGenerator
   
   # Train model
   pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
   pipeline.fit(train_df, target_col='default')
   
   # Generate comprehensive development report
   report_gen = HTMLReportGenerator()
   
   report_html = report_gen.generate(
       pipeline=pipeline,
       test_df=test_df,
       target_col='default',
       report_title="Credit Scorecard Development Report",
       author="Data Science Team",
       include_plots=True,
       include_binning=True,
       include_performance=True,
       include_summary=True
   )
   
   # Save for stakeholders
   with open('dev_report.html', 'w', encoding='utf-8') as f:
       f.write(report_html)

Production Phase
~~~~~~~~~~~~~~~~

Use ObservabilityDashboard for ongoing production monitoring:

.. code-block:: python

   from cr_score.reporting import ObservabilityDashboard
   import schedule
   import time
   
   def generate_daily_dashboard():
       """Generate daily monitoring dashboard."""
       
       # Collect data from monitors
       perf_metrics = perf_monitor.get_metrics_summary(window_size=30)
       health = perf_monitor.check_health()
       
       drift_report = drift_monitor.detect_drift(X_recent)
       pred_stats = pred_monitor.get_prediction_summary()
       
       metrics = metrics_collector.get_metrics()
       alerts = alert_manager.get_active_alerts()
       alert_summary = alert_manager.get_alert_summary()
       
       # Create dashboard
       dashboard = ObservabilityDashboard(
           title=f"Production Monitoring - {datetime.now().strftime('%Y-%m-%d')}"
       )
       
       dashboard.add_performance_section(perf_metrics, health)
       dashboard.add_drift_section(drift_report)
       dashboard.add_prediction_section(pred_stats)
       dashboard.add_metrics_section(metrics)
       dashboard.add_alerts_section(alerts, alert_summary)
       
       # Export
       dashboard.export(f'dashboards/dashboard_{datetime.now().strftime("%Y%m%d")}.html')
   
   # Schedule daily generation
   schedule.every().day.at("09:00").do(generate_daily_dashboard)

Best Practices
--------------

HTML Reports
~~~~~~~~~~~~

1. **Include All Stakeholder-Relevant Sections**
   
   Customize for audience:
   
   .. code-block:: python
   
      # For technical team
      report_gen.generate(
          include_binning=True,
          include_coefficients=True,
          include_diagnostics=True
      )
      
      # For business stakeholders
      report_gen.generate(
          include_summary=True,
          include_performance=True,
          include_binning=False,  # Too technical
          include_coefficients=False
      )

2. **Version Control Reports**
   
   Save reports with timestamps:
   
   .. code-block:: python
   
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f'report_{model_version}_{timestamp}.html'

3. **Professional Styling**
   
   Reports include professional CSS and interactive Plotly charts

Observability Dashboards
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Real-Time Updates**
   
   Refresh dashboards regularly:
   
   .. code-block:: python
   
      # Hourly updates
      schedule.every().hour.do(generate_dashboard)

2. **Historical Archives**
   
   Keep dashboard history:
   
   .. code-block:: python
   
      # Save with date
      dashboard.export(f'dashboards/dashboard_{date}.html')

3. **Color-Coded Status**
   
   Dashboards use traffic light colors:
   - 🟢 Green: Healthy/Stable
   - 🟡 Orange: Warning
   - 🔴 Red: Critical

4. **Interactive Elements**
   
   Use Plotly for interactive charts when possible

Combining Reports and Dashboards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Development: One-time comprehensive report
   dev_report = report_gen.generate(pipeline, test_df, 'default')
   with open('model_documentation.html', 'w') as f:
       f.write(dev_report)
   
   # Production: Continuous monitoring dashboard
   def update_monitoring():
       dashboard = ObservabilityDashboard()
       # ... add sections ...
       dashboard.export('current_monitoring.html')
   
   schedule.every(15).minutes.do(update_monitoring)

Export Formats
--------------

HTML Format
~~~~~~~~~~~

Default format with full interactivity:

.. code-block:: python

   # HTML reports
   report_gen.generate(..., format='html')
   
   # HTML dashboards
   dashboard.export('dashboard.html')

PDF Export
~~~~~~~~~~

Convert HTML to PDF using external tools:

.. code-block:: python

   import pdfkit
   
   # Generate HTML
   html_report = report_gen.generate(...)
   
   # Convert to PDF
   pdfkit.from_string(html_report, 'report.pdf')

Excel Export
~~~~~~~~~~~~

Export data tables to Excel:

.. code-block:: python

   import pandas as pd
   
   # Get summary data
   summary = pipeline.get_summary()
   
   # Export to Excel
   with pd.ExcelWriter('scorecard_summary.xlsx') as writer:
       pd.DataFrame(summary['iv_summary']).to_excel(
           writer, sheet_name='Feature Importance', index=False
       )
       metrics_df.to_excel(writer, sheet_name='Performance', index=False)

Integration Examples
--------------------

With MLflow
~~~~~~~~~~~

.. code-block:: python

   import mlflow
   
   with mlflow.start_run():
       # Train model
       pipeline.fit(train_df, target_col='default')
       
       # Generate report
       report_html = report_gen.generate(pipeline, test_df, 'default')
       
       # Log as artifact
       with open('report.html', 'w') as f:
           f.write(report_html)
       mlflow.log_artifact('report.html')

With Cloud Storage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import boto3
   
   # Generate dashboard
   dashboard.export('temp_dashboard.html')
   
   # Upload to S3
   s3 = boto3.client('s3')
   s3.upload_file(
       'temp_dashboard.html',
       'my-bucket',
       'dashboards/latest.html',
       ExtraArgs={'ContentType': 'text/html'}
   )

See Also
--------

- :doc:`/api/monitoring` - Monitoring and observability
- :doc:`/api/viz` - Visualization functions
- :doc:`/api/pipeline` - ScorecardPipeline
- :doc:`/guides/reporting` - Reporting guide
