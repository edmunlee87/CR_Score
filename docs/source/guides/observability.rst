Observability & Monitoring Guide
==================================

Complete guide for production monitoring and observability.

Overview
--------

Production scorecards require comprehensive monitoring to ensure:

- **Model Performance**: Maintain predictive accuracy
- **Data Quality**: Detect distribution shifts
- **Operational Health**: Track system metrics
- **Compliance**: Audit trails and explainability

This guide covers setting up end-to-end observability.

Quick Start
-----------

Basic Monitoring Setup
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.monitoring import PerformanceMonitor, DriftMonitor
   
   # Train model
   pipeline = ScorecardPipeline()
   pipeline.fit(train_df, target_col='default')
   
   # Set up monitors
   perf_monitor = PerformanceMonitor(
       baseline_metrics={'auc': 0.85},
       alert_thresholds={'auc': 0.05}
   )
   
   drift_monitor = DriftMonitor(
       reference_data=train_df,
       psi_threshold=0.1
   )
   
   # In production: monitor predictions
   scores = pipeline.predict(production_df)
   probas = pipeline.predict_proba(production_df)
   
   # Check performance (when labels available)
   metrics = perf_monitor.record_predictions(y_true, y_pred, probas)
   
   # Check drift
   drift_report = drift_monitor.detect_drift(production_df)
   
   # Generate dashboard
   from cr_score.reporting import ObservabilityDashboard
   dashboard = ObservabilityDashboard()
   dashboard.add_performance_section(metrics, perf_monitor.check_health())
   dashboard.add_drift_section(drift_report)
   dashboard.export('monitoring.html')

Performance Monitoring
----------------------

Setting Baselines
~~~~~~~~~~~~~~~~~

Always use validation/test data for baselines, not training data:

.. code-block:: python

   from cr_score.monitoring import PerformanceMonitor
   from sklearn.metrics import roc_auc_score, precision_score, recall_score
   
   # Calculate baseline metrics on validation set
   val_probas = pipeline.predict_proba(X_val)
   val_pred = (val_probas > 0.5).astype(int)
   
   baseline_metrics = {
       'auc': roc_auc_score(y_val, val_probas),
       'precision': precision_score(y_val, val_pred),
       'recall': recall_score(y_val, val_pred)
   }
   
   # Set alert thresholds (% degradation)
   alert_thresholds = {
       'auc': 0.05,      # Alert if AUC drops >5%
       'precision': 0.10,  # Alert if precision drops >10%
       'recall': 0.10
   }
   
   # Initialize monitor
   monitor = PerformanceMonitor(
       baseline_metrics=baseline_metrics,
       alert_thresholds=alert_thresholds,
       storage_path='./monitoring_data'
   )

Recording Predictions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In production scoring loop
   for batch in production_batches:
       # Score batch
       probas = pipeline.predict_proba(batch.features)
       pred = (probas > threshold).astype(int)
       
       # When labels become available (e.g., after 3 months)
       if batch.has_labels:
           metrics = monitor.record_predictions(
               y_true=batch.labels,
               y_pred=pred,
               y_proba=probas,
               metadata={
                   'batch_id': batch.id,
                   'date': batch.date,
                   'model_version': 'v2.0'
               }
           )
           
           print(f"Batch {batch.id} metrics:")
           print(f"  AUC: {metrics['auc']:.3f}")
           print(f"  Precision: {metrics['precision']:.3f}")
           print(f"  Recall: {metrics['recall']:.3f}")

Health Checks
~~~~~~~~~~~~~

.. code-block:: python

   # Check model health
   health = monitor.check_health()
   
   if health['status'] == 'critical':
       print("⚠️  CRITICAL: Model performance degraded!")
       for alert in health['alerts']:
           print(f"  {alert['metric']}: {alert['current']:.3f} " 
                 f"(baseline: {alert['baseline']:.3f}, "
                 f"drop: {alert['degradation_pct']:.1%})")
       
       # Take action: retrain, investigate, rollback
       trigger_model_retrain()
   
   elif health['status'] == 'warning':
       print("⚡ WARNING: Performance declining")
       schedule_investigation()
   
   else:
       print("✅ HEALTHY: Model performing as expected")

Visualizing Trends
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot metrics over time
   monitor.plot_metrics_over_time(
       metrics=['auc', 'precision', 'recall'],
       window_size=30  # 30-day rolling window
   )
   
   # Export report
   monitor.export_report('performance_report.html', format='html')

Drift Detection
---------------

Understanding Drift Types
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Covariate Shift**: Input feature distributions change
2. **Prior Shift**: Target distribution changes
3. **Concept Drift**: Relationship between inputs and target changes

We primarily monitor covariate shift using PSI and KS tests.

Setting Up Drift Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import DriftMonitor
   
   # Use training data as reference
   monitor = DriftMonitor(
       reference_data=X_train,
       psi_threshold=0.1,   # PSI > 0.1 triggers warning
       ks_threshold=0.05,   # KS p-value < 0.05 is significant
       storage_path='./monitoring_data/drift'
   )

Detecting Drift
~~~~~~~~~~~~~~~

.. code-block:: python

   # Check production batch for drift
   drift_report = monitor.detect_drift(X_production)
   
   print(f"Drift Status: {drift_report['overall_status']}")
   print(f"  Critical: {drift_report['drift_summary']['critical']} features")
   print(f"  Warning:  {drift_report['drift_summary']['warning']} features")
   print(f"  Stable:   {drift_report['drift_summary']['stable']} features")
   
   # Inspect drifted features
   for feat, results in drift_report['feature_results'].items():
       if results['status'] in ['critical', 'warning']:
           print(f"\n{feat}:")
           print(f"  PSI: {results['psi']:.3f}")
           print(f"  KS statistic: {results['ks_statistic']:.3f}")
           print(f"  KS p-value: {results['ks_pvalue']:.4f}")

Visualizing Drift
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot drift summary
   monitor.plot_drift_summary(drift_report, top_n=20)
   
   # Export detailed drift report
   monitor.export_drift_report(drift_report, 'drift_report.html', format='html')

Handling Drift
~~~~~~~~~~~~~~

When drift is detected:

1. **Investigate Root Cause**
   
   .. code-block:: python
   
      # Compare distributions
      import matplotlib.pyplot as plt
      
      for feature in drifted_features:
          plt.figure(figsize=(10, 4))
          plt.subplot(1, 2, 1)
          X_train[feature].hist(bins=50, alpha=0.5, label='Training')
          X_production[feature].hist(bins=50, alpha=0.5, label='Production')
          plt.legend()
          plt.title(f'{feature} Distribution')
          plt.show()

2. **Decide on Action**
   
   - **Minor Drift**: Continue monitoring
   - **Moderate Drift**: Investigate data pipeline
   - **Severe Drift**: Consider model retraining

3. **Update Reference if Legitimate**
   
   .. code-block:: python
   
      # If drift is due to legitimate market changes
      monitor.update_reference_data(X_new_baseline)

Prediction Monitoring
---------------------

Track Score Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import PredictionMonitor
   
   monitor = PredictionMonitor(
       expected_score_range=(300, 850),
       expected_proba_range=(0.0, 1.0),
       storage_path='./monitoring_data/predictions'
   )
   
   # Record predictions
   stats = monitor.record_predictions(
       scores=scores,
       probabilities=probas,
       metadata={'timestamp': datetime.now()}
   )
   
   print(f"Score distribution:")
   print(f"  Mean: {stats['score_mean']:.1f}")
   print(f"  Std: {stats['score_std']:.1f}")
   print(f"  Min: {stats['score_min']:.1f}")
   print(f"  Max: {stats['score_max']:.1f}")

Anomaly Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect anomalous predictions
   anomalies = monitor.detect_anomalies(
       scores=scores,
       method='iqr',  # or 'zscore'
       threshold=1.5
   )
   
   if anomalies['n_anomalies'] > 0:
       print(f"⚠️  {anomalies['n_anomalies']} anomalous predictions!")
       print(f"   Rate: {anomalies['pct_anomalies']:.2f}%")
       
       # Inspect anomalies
       anomaly_indices = anomalies['anomaly_indices']
       anomalous_scores = scores[anomaly_indices]
       print(f"   Score range: {anomalous_scores.min():.1f} - {anomalous_scores.max():.1f}")

Alert Management
----------------

Setting Up Alerts
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import AlertManager, AlertSeverity
   
   # Define notification handlers
   def email_handler(alert):
       send_email(
           to='ml-team@company.com',
           subject=f"[{alert['severity']}] {alert['title']}",
           body=str(alert['details'])
       )
   
   def slack_handler(alert):
       if alert['severity'] == 'critical':
           channel = '#ml-critical'
       else:
           channel = '#ml-alerts'
       
       post_to_slack(channel, alert)
   
   # Initialize manager
   alert_manager = AlertManager(
       storage_path='./monitoring_data/alerts',
       notification_handlers=[email_handler, slack_handler]
   )

Creating Alerts
~~~~~~~~~~~~~~~

.. code-block:: python

   # Automatically create alerts from monitors
   health = perf_monitor.check_health()
   
   if health['status'] == 'critical':
       for alert_info in health['alerts']:
           alert = alert_manager.create_alert(
               title=f"Performance Degradation: {alert_info['metric']}",
               severity=AlertSeverity.CRITICAL,
               details=alert_info,
               source='performance_monitor'
           )

Managing Alerts
~~~~~~~~~~~~~~~

.. code-block:: python

   # Get active alerts
   critical_alerts = alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
   
   for alert in critical_alerts:
       print(f"{alert['severity']}: {alert['title']}")
       print(f"  Created: {alert['created_at']}")
       print(f"  Source: {alert['source']}")
   
   # Acknowledge alert
   alert_manager.acknowledge_alert(
       alert['alert_id'],
       acknowledged_by='john.doe@company.com'
   )
   
   # Resolve alert
   alert_manager.resolve_alert(
       alert['alert_id'],
       resolution_notes='Model retrained with recent data. New AUC: 0.86'
   )
   
   # Get alert summary
   summary = alert_manager.get_alert_summary()
   print(f"Alert Summary:")
   print(f"  Total: {summary['total']}")
   print(f"  Active: {summary['active']}")
   print(f"  Resolved: {summary['resolved']}")

Metrics Collection
------------------

Prometheus-Compatible Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import MetricsCollector
   from cr_score.monitoring.metrics_collector import register_standard_metrics
   
   # Initialize collector
   collector = MetricsCollector(
       storage_path='./monitoring_data/metrics',
       enable_prometheus=True
   )
   
   # Register standard metrics
   register_standard_metrics(collector)

Recording Metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Counters (monotonically increasing)
   collector.increment_counter('predictions_total', value=100)
   collector.increment_counter('predictions_total', value=1, labels={'model': 'v2'})
   
   # Gauges (can go up or down)
   collector.set_gauge('model_auc', value=0.850)
   collector.set_gauge('active_alerts', value=len(critical_alerts))
   
   # Histograms (for distributions)
   collector.record_histogram('prediction_latency_ms', value=45.2)
   collector.record_histogram('score_value', value=650)
   
   # Timers (for measuring durations)
   with collector.time_operation('batch_scoring'):
       scores = pipeline.predict(X_batch)

Exporting Metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prometheus format
   prometheus_metrics = collector.export_prometheus()
   
   # Expose via HTTP endpoint for Prometheus scraping
   from flask import Flask, Response
   
   app = Flask(__name__)
   
   @app.route('/metrics')
   def metrics():
       return Response(collector.export_prometheus(), mimetype='text/plain')
   
   # Or export to file
   collector.export_json('metrics.json')

Observability Dashboard
-----------------------

Complete Dashboard
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.reporting import ObservabilityDashboard
   from datetime import datetime
   
   def generate_dashboard():
       """Generate comprehensive monitoring dashboard."""
       
       # Create dashboard
       dashboard = ObservabilityDashboard(
           title=f"Production Scorecard - {datetime.now().strftime('%Y-%m-%d')}"
       )
       
       # Section 1: Performance
       metrics_df = perf_monitor.get_metrics_summary(window_size=30)
       health = perf_monitor.check_health()
       dashboard.add_performance_section(metrics_df, health)
       
       # Section 2: Drift
       drift_report = drift_monitor.detect_drift(X_recent)
       dashboard.add_drift_section(drift_report)
       
       # Section 3: Predictions
       pred_stats = pred_monitor.get_prediction_summary()
       dashboard.add_prediction_section(pred_stats)
       
       # Section 4: System Metrics
       metrics = collector.get_metrics()
       dashboard.add_metrics_section(metrics)
       
       # Section 5: Alerts
       active_alerts = alert_manager.get_active_alerts()
       alert_summary = alert_manager.get_alert_summary()
       dashboard.add_alerts_section(active_alerts, alert_summary)
       
       # Export
       dashboard.export('monitoring_dashboard.html')
       print("✅ Dashboard generated: monitoring_dashboard.html")

Automated Dashboard Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import schedule
   import time
   
   # Generate dashboard every hour
   schedule.every().hour.do(generate_dashboard)
   
   # Run scheduler
   while True:
       schedule.run_pending()
       time.sleep(60)

Best Practices
--------------

1. Establish Baselines Early
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use validation data for baselines
- Update baselines quarterly or after retraining
- Document baseline assumptions

2. Set Appropriate Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Start conservative (stricter thresholds)
- Adjust based on operational experience
- Different thresholds for different features

3. Monitor Regularly
~~~~~~~~~~~~~~~~~~~~

- Daily drift checks
- Weekly performance reviews
- Monthly deep dives

4. Automate Responses
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def auto_response(health):
       if health['status'] == 'critical':
           # Automatic actions
           alert_manager.create_alert(...)
           trigger_investigation()
           consider_rollback()
       elif health['status'] == 'warning':
           # Schedule review
           schedule_performance_review()

5. Maintain Alert Hygiene
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Acknowledge alerts promptly
- Document resolutions
- Avoid alert fatigue with proper thresholds

6. Version Everything
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metadata = {
       'model_version': 'v2.0',
       'data_version': '2026-01',
       'config_hash': config_hash,
       'timestamp': datetime.now()
   }

Production Deployment Pattern
------------------------------

Complete Production Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.monitoring import *
   from cr_score.reporting import ObservabilityDashboard
   import logging
   
   # Set up logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   # Load model
   pipeline = ScorecardPipeline.load('production_model.pkl')
   logger.info(f"Loaded model version: {pipeline.version}")
   
   # Initialize monitoring
   perf_monitor = PerformanceMonitor(
       baseline_metrics={'auc': 0.850},
       alert_thresholds={'auc': 0.05}
   )
   
   drift_monitor = DriftMonitor(reference_data=X_train, psi_threshold=0.1)
   pred_monitor = PredictionMonitor(expected_score_range=(300, 850))
   alert_manager = AlertManager()
   metrics_collector = MetricsCollector(enable_prometheus=True)
   
   # Production scoring function
   def score_and_monitor(X_batch, y_batch=None):
       """Score batch and perform monitoring."""
       
       # Time scoring
       with metrics_collector.time_operation('prediction'):
           scores = pipeline.predict(X_batch)
           probas = pipeline.predict_proba(X_batch)
       
       # Record predictions
       metrics_collector.increment_counter('predictions_total', value=len(X_batch))
       pred_stats = pred_monitor.record_predictions(scores, probas)
       
       # Check drift
       drift_report = drift_monitor.detect_drift(X_batch)
       
       if drift_report['overall_status'] == 'critical':
           alert_manager.create_alert(
               title='Critical Data Drift',
               severity=AlertSeverity.CRITICAL,
               details=drift_report['drift_summary']
           )
       
       # Performance monitoring (when labels available)
       if y_batch is not None:
           y_pred = (probas > 0.5).astype(int)
           perf_metrics = perf_monitor.record_predictions(y_batch, y_pred, probas)
           
           health = perf_monitor.check_health()
           
           if health['status'] == 'critical':
               for alert_info in health['alerts']:
                   alert_manager.create_alert(
                       title=f"Performance Degradation: {alert_info['metric']}",
                       severity=AlertSeverity.CRITICAL,
                       details=alert_info
                   )
       
       return scores
   
   # Schedule dashboard generation
   schedule.every().hour.do(generate_dashboard)
   
   logger.info("✅ Production monitoring initialized")

See Also
--------

- :doc:`/api/monitoring` - Monitoring API reference
- :doc:`/api/explainability` - Explainability tools
- :doc:`/api/reporting` - Dashboard generation
- :doc:`/api/templates` - Configuration templates
