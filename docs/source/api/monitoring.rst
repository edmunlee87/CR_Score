Monitoring & Observability
===========================

Production monitoring and observability for credit scorecards.

Overview
--------

The monitoring module provides comprehensive observability for production scorecards:

- **Performance Monitoring**: Track metrics over time, detect degradation
- **Drift Detection**: Monitor distribution shifts with PSI/KS tests
- **Prediction Monitoring**: Track score distributions and anomalies
- **Alert Management**: Multi-severity alerting with notification routing
- **Metrics Collection**: Prometheus-compatible metrics and counters

Key Features
------------

✅ **Real-Time Monitoring**
   - Performance metrics tracking
   - Automated degradation detection
   - Historical trend analysis

✅ **Drift Detection**
   - Population Stability Index (PSI)
   - Kolmogorov-Smirnov tests
   - Per-feature drift tracking

✅ **Production Observability**
   - Prometheus metrics export
   - Custom metrics collection
   - Interactive dashboards

✅ **Intelligent Alerting**
   - Multi-severity levels
   - Configurable thresholds
   - Notification routing

Performance Monitor
-------------------

.. automodule:: cr_score.monitoring.performance_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import PerformanceMonitor
   
   # Initialize with baseline
   monitor = PerformanceMonitor(
       baseline_metrics={'auc': 0.85, 'precision': 0.75},
       alert_thresholds={'auc': 0.05, 'precision': 0.10},
       storage_path='./monitoring_data'
   )
   
   # Record predictions (in production)
   metrics = monitor.record_predictions(
       y_true=y_true,
       y_pred=y_pred,
       y_proba=y_proba,
       metadata={'batch_id': 'batch_001', 'date': '2026-01-16'}
   )
   
   # Check health status
   health = monitor.check_health()
   
   if health['status'] == 'critical':
       print(f"ALERT: {len(health['alerts'])} critical issues detected!")
       for alert in health['alerts']:
           print(f"  {alert['metric']}: {alert['degradation_pct']:.1%} degradation")
   
   # Plot metrics over time
   monitor.plot_metrics_over_time(metrics=['auc', 'precision', 'recall'])
   
   # Export report
   monitor.export_report('performance_report.html', format='html')

Drift Monitor
-------------

.. automodule:: cr_score.monitoring.drift_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import DriftMonitor
   
   # Initialize with reference (training) data
   monitor = DriftMonitor(
       reference_data=X_train,
       psi_threshold=0.1,  # PSI > 0.1 = warning
       ks_threshold=0.05,  # KS p-value < 0.05 = significant
       storage_path='./monitoring_data'
   )
   
   # Detect drift in production data
   drift_report = monitor.detect_drift(X_production)
   
   # Check drift status
   print(f"Drift Status: {drift_report['overall_status']}")
   print(f"  Critical: {drift_report['drift_summary']['critical']}")
   print(f"  Warning:  {drift_report['drift_summary']['warning']}")
   print(f"  Stable:   {drift_report['drift_summary']['stable']}")
   
   # Visualize drift
   monitor.plot_drift_summary(drift_report, top_n=20)
   
   # Export drift report
   monitor.export_drift_report(drift_report, 'drift_report.html', format='html')

Prediction Monitor
------------------

.. automodule:: cr_score.monitoring.prediction_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import PredictionMonitor
   
   # Initialize with expected ranges
   monitor = PredictionMonitor(
       expected_score_range=(300, 850),
       expected_proba_range=(0.0, 1.0),
       storage_path='./monitoring_data'
   )
   
   # Record predictions
   stats = monitor.record_predictions(
       scores=scores,
       probabilities=probabilities,
       metadata={'timestamp': '2026-01-16T10:00:00'}
   )
   
   # Detect anomalies
   anomalies = monitor.detect_anomalies(
       scores=scores,
       method='iqr',
       threshold=1.5
   )
   
   if anomalies['n_anomalies'] > 0:
       print(f"WARNING: {anomalies['n_anomalies']} anomalous predictions detected!")
       print(f"Anomaly rate: {anomalies['pct_anomalies']:.2f}%")
   
   # Visualize distribution
   monitor.plot_prediction_distribution(scores, title='Production Scores')

Alert Manager
-------------

.. automodule:: cr_score.monitoring.alert_manager
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.monitoring import AlertManager, AlertSeverity
   from cr_score.monitoring.alert_manager import (
       email_notification_handler,
       slack_notification_handler
   )
   
   # Initialize with notification handlers
   manager = AlertManager(
       storage_path='./monitoring_data/alerts',
       notification_handlers=[
           email_notification_handler,
           slack_notification_handler
       ]
   )
   
   # Create alerts
   alert = manager.create_alert(
       title='Model Performance Degradation',
       severity=AlertSeverity.CRITICAL,
       details={
           'metric': 'auc',
           'baseline': 0.850,
           'current': 0.780,
           'degradation': 0.070
       },
       source='performance_monitor'
   )
   
   # Acknowledge alert
   manager.acknowledge_alert(alert['alert_id'])
   
   # Resolve alert
   manager.resolve_alert(
       alert['alert_id'],
       resolution_notes='Model retrained with fresh data'
   )
   
   # Get active alerts
   critical_alerts = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
   
   # Export alerts
   manager.export_alerts('alerts.json', format='json')

Metrics Collector
-----------------

.. automodule:: cr_score.monitoring.metrics_collector
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
~~~~~~~~~~~~~

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
   
   # Increment counters
   collector.increment_counter('predictions_total', value=100)
   collector.increment_counter('predictions_total', value=1, labels={'model': 'v2'})
   
   # Set gauges
   collector.set_gauge('model_auc', value=0.850)
   collector.set_gauge('active_models', value=1)
   
   # Record histograms
   collector.record_histogram('prediction_latency_ms', value=45.2)
   collector.record_histogram('score_value', value=650)
   
   # Time operations
   with collector.time_operation('model_prediction'):
       scores = model.predict(X)
   
   # Get metrics
   metrics = collector.get_metrics()
   
   # Export in Prometheus format
   prometheus_metrics = collector.export_prometheus()
   print(prometheus_metrics)
   
   # Export to JSON
   collector.export_json('metrics.json')

Production Deployment
---------------------

Complete Production Monitoring Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.monitoring import (
       PerformanceMonitor,
       DriftMonitor,
       PredictionMonitor,
       AlertManager,
       MetricsCollector,
       AlertSeverity
   )
   
   # Load production model
   pipeline = ScorecardPipeline.load('production_model.pkl')
   
   # Initialize monitors
   perf_monitor = PerformanceMonitor(
       baseline_metrics={'auc': 0.850},
       alert_thresholds={'auc': 0.05}
   )
   
   drift_monitor = DriftMonitor(
       reference_data=X_train,
       psi_threshold=0.1
   )
   
   pred_monitor = PredictionMonitor(
       expected_score_range=(300, 850)
   )
   
   alert_manager = AlertManager()
   metrics_collector = MetricsCollector(enable_prometheus=True)
   
   # Production prediction loop
   def score_batch(X_batch, y_batch_true=None):
       """Score a batch and monitor."""
       
       # Time prediction
       with metrics_collector.time_operation('prediction'):
           scores = pipeline.predict(X_batch)
           probas = pipeline.predict_proba(X_batch)
       
       # Record metrics
       metrics_collector.increment_counter('predictions_total', value=len(X_batch))
       
       # Monitor predictions
       pred_stats = pred_monitor.record_predictions(scores, probas)
       
       # Detect drift
       drift_report = drift_monitor.detect_drift(X_batch)
       
       if drift_report['overall_status'] == 'critical':
           alert_manager.create_alert(
               title='Critical Data Drift Detected',
               severity=AlertSeverity.CRITICAL,
               details=drift_report['drift_summary']
           )
       
       # If labels available, monitor performance
       if y_batch_true is not None:
           y_pred = (probas > 0.5).astype(int)
           perf_metrics = perf_monitor.record_predictions(
               y_batch_true,
               y_pred,
               probas
           )
           
           # Check health
           health = perf_monitor.check_health()
           
           if health['status'] == 'critical':
               for alert in health['alerts']:
                   alert_manager.create_alert(
                       title=f"Performance Degradation: {alert['metric']}",
                       severity=AlertSeverity.CRITICAL,
                       details=alert
                   )
       
       return scores

Monitoring Dashboard
~~~~~~~~~~~~~~~~~~~~

Create observability dashboard:

.. code-block:: python

   from cr_score.reporting import ObservabilityDashboard
   
   # Create dashboard
   dashboard = ObservabilityDashboard(
       title="Production Scorecard Monitoring"
   )
   
   # Add performance section
   metrics_df = perf_monitor.get_metrics_summary()
   health = perf_monitor.check_health()
   dashboard.add_performance_section(metrics_df, health)
   
   # Add drift section
   dashboard.add_drift_section(drift_report)
   
   # Add prediction section
   pred_summary = pred_monitor.get_prediction_summary()
   dashboard.add_prediction_section(pred_summary)
   
   # Add metrics section
   dashboard.add_metrics_section(metrics_collector.get_metrics())
   
   # Add alerts section
   active_alerts = alert_manager.get_active_alerts()
   alert_summary = alert_manager.get_alert_summary()
   dashboard.add_alerts_section(active_alerts, alert_summary)
   
   # Export dashboard
   dashboard.export('monitoring_dashboard.html')

Best Practices
--------------

Setting Baselines
~~~~~~~~~~~~~~~~~

1. **Use Validation Data for Baselines**
   
   Don't use training data:
   
   .. code-block:: python
   
      # Calculate baseline on validation/test set
      val_probas = model.predict_proba(X_val)[:, 1]
      baseline_auc = roc_auc_score(y_val, val_probas)
      
      monitor = PerformanceMonitor(
          baseline_metrics={'auc': baseline_auc}
      )

2. **Conservative Thresholds**
   
   Start with conservative thresholds:
   
   .. code-block:: python
   
      alert_thresholds = {
          'auc': 0.05,      # 5% drop
          'precision': 0.10,  # 10% drop
          'recall': 0.10
      }

Drift Detection
~~~~~~~~~~~~~~~

1. **Monitor All Features**
   
   Track drift for all model inputs:
   
   .. code-block:: python
   
      drift_report = monitor.detect_drift(X_production, features=None)  # All features

2. **Different Thresholds by Feature Importance**
   
   Stricter thresholds for important features:
   
   .. code-block:: python
   
      for feat in important_features:
          if drift_results[feat]['psi'] > 0.1:
              # Alert!

3. **Regular Cadence**
   
   Check drift daily/weekly depending on volume

Alert Configuration
~~~~~~~~~~~~~~~~~~~

1. **Severity Levels**
   
   - INFO: Informational, no action needed
   - WARNING: Investigate soon
   - CRITICAL: Immediate action required

2. **Notification Routing**
   
   Route by severity:
   
   .. code-block:: python
   
      def custom_notification_handler(alert):
          if alert['severity'] == 'critical':
              send_pagerduty(alert)
          elif alert['severity'] == 'warning':
              send_slack(alert)
          else:
              log_to_file(alert)

3. **Alert Fatigue**
   
   Avoid too many alerts:
   - Set appropriate thresholds
   - Batch similar alerts
   - Implement cooldown periods

See Also
--------

- :doc:`/api/reporting` - ObservabilityDashboard
- :doc:`/api/pipeline` - ScorecardPipeline
- :doc:`/guides/quickstart` - Quick start guide
- :doc:`/api/explainability` - Model explainability
