"""
Metrics collector for observability.

Collects and exposes metrics in formats compatible with Prometheus, CloudWatch, etc.
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cr_score.core.logging import get_audit_logger


class MetricsCollector:
    """
    Collect and expose observability metrics.
    
    Provides metrics in multiple formats for monitoring systems.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.increment_counter("predictions_total")
        >>> collector.record_histogram("prediction_latency_ms", 45.2)
        >>> collector.set_gauge("active_models", 1)
        >>> metrics = collector.get_metrics()
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_prometheus: bool = True,
    ) -> None:
        """
        Initialize metrics collector.
        
        Args:
            storage_path: Path to store metrics
            enable_prometheus: Enable Prometheus format export
        """
        self.storage_path = Path(storage_path) if storage_path else Path('./monitoring_data/metrics')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_prometheus = enable_prometheus
        self.logger = get_audit_logger()
        
        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metadata
        self.metric_metadata: Dict[str, Dict[str, str]] = {}
        self.start_time = time.time()
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        self.logger.debug(f"Counter incremented: {key} += {value}")
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        self.logger.debug(f"Gauge set: {key} = {value}")
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a histogram observation.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        self.logger.debug(f"Histogram recorded: {key} = {value}")
    
    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager to time operations.
        
        Args:
            name: Metric name
            labels: Metric labels
        
        Example:
            >>> with collector.time_operation("model_prediction"):
            ...     model.predict(X)
        """
        return TimerContext(self, name, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and labels."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def register_metric(
        self,
        name: str,
        metric_type: str,
        description: str,
        unit: Optional[str] = None,
    ) -> None:
        """
        Register metric metadata.
        
        Args:
            name: Metric name
            metric_type: Type (counter, gauge, histogram, timer)
            description: Metric description
            unit: Unit of measurement
        """
        self.metric_metadata[name] = {
            'type': metric_type,
            'description': description,
            'unit': unit or '',
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': self._summarize_histograms(),
            'timers': self._summarize_timers(),
        }
        
        return metrics
    
    def _summarize_histograms(self) -> Dict[str, Dict[str, float]]:
        """Summarize histogram data."""
        import numpy as np
        
        summaries = {}
        for key, values in self.histograms.items():
            if values:
                summaries[key] = {
                    'count': len(values),
                    'sum': float(np.sum(values)),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p50': float(np.percentile(values, 50)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99)),
                }
        
        return summaries
    
    def _summarize_timers(self) -> Dict[str, Dict[str, float]]:
        """Summarize timer data."""
        return self._summarize_histograms()  # Same logic
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            lines.append(f"{key} {value}")
        
        # Histograms (as summaries)
        for key, summary in self._summarize_histograms().items():
            lines.append(f"{key}_count {summary['count']}")
            lines.append(f"{key}_sum {summary['sum']}")
            lines.append(f"{key}{{quantile=\"0.5\"}} {summary['p50']}")
            lines.append(f"{key}{{quantile=\"0.95\"}} {summary['p95']}")
            lines.append(f"{key}{{quantile=\"0.99\"}} {summary['p99']}")
        
        return "\n".join(lines)
    
    def export_json(self, filepath: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        metrics = self.get_metrics()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()
        self.start_time = time.time()
        
        self.logger.info("Metrics reset")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]],
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        key = self.collector._make_key(self.name, self.labels)
        self.collector.timers[key].append(duration_ms)


# Standard metrics registration
def register_standard_metrics(collector: MetricsCollector) -> None:
    """Register standard scorecard metrics."""
    
    # Prediction metrics
    collector.register_metric(
        "predictions_total",
        "counter",
        "Total number of predictions made"
    )
    
    collector.register_metric(
        "prediction_latency_ms",
        "histogram",
        "Prediction latency in milliseconds",
        "ms"
    )
    
    collector.register_metric(
        "model_auc",
        "gauge",
        "Current model AUC score"
    )
    
    collector.register_metric(
        "drift_psi",
        "gauge",
        "Population Stability Index for drift"
    )
    
    collector.register_metric(
        "alerts_active",
        "gauge",
        "Number of active alerts"
    )
    
    collector.register_metric(
        "data_quality_score",
        "gauge",
        "Data quality score (0-1)"
    )
