"""
Performance monitoring for production scorecards.

Tracks model performance metrics over time with alerting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from cr_score.core.logging import get_audit_logger


class PerformanceMonitor:
    """
    Monitor model performance in production.
    
    Tracks metrics over time, detects degradation, and triggers alerts.
    
    Example:
        >>> monitor = PerformanceMonitor(baseline_auc=0.85, alert_threshold=0.05)
        >>> monitor.record_predictions(y_true, y_pred, y_proba)
        >>> status = monitor.check_health()
    """
    
    def __init__(
        self,
        baseline_metrics: Optional[Dict[str, float]] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline metrics from training/validation
            alert_thresholds: Thresholds for alerting (% degradation)
            storage_path: Path to store monitoring data
        """
        self.baseline_metrics = baseline_metrics or {}
        self.alert_thresholds = alert_thresholds or {
            'auc': 0.05,  # 5% drop triggers alert
            'precision': 0.10,
            'recall': 0.10,
            'f1': 0.10,
        }
        self.storage_path = Path(storage_path) if storage_path else Path('./monitoring_data')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_audit_logger()
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Load existing history
        self._load_history()
    
    def _load_history(self) -> None:
        """Load monitoring history from disk."""
        history_file = self.storage_path / 'performance_history.jsonl'
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.metrics_history = [json.loads(line) for line in f]
            self.logger.info(
                f"Loaded {len(self.metrics_history)} historical metrics"
            )
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to disk."""
        history_file = self.storage_path / 'performance_history.jsonl'
        with open(history_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def record_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Record predictions and calculate metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            metadata: Additional metadata (e.g., batch_id, date)
            threshold: Classification threshold
        
        Returns:
            Calculated metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate metrics
        metrics = {
            'timestamp': timestamp,
            'n_samples': len(y_true),
            'n_positive': int(y_true.sum()),
            'n_predicted_positive': int(y_pred.sum()),
            'actual_rate': float(y_true.mean()),
            'predicted_rate': float(y_pred.mean()),
        }
        
        # Classification metrics
        try:
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        except Exception as e:
            self.logger.warning(f"Error calculating classification metrics: {e}")
        
        # Probabilistic metrics
        if y_proba is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_proba))
            except Exception as e:
                self.logger.warning(f"Error calculating AUC: {e}")
        
        # Add metadata
        if metadata:
            metrics.update(metadata)
        
        # Save metrics
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        self.logger.info(
            "Recorded performance metrics",
            n_samples=metrics['n_samples'],
            auc=metrics.get('auc'),
        )
        
        return metrics
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check model health status.
        
        Returns:
            Health status with alerts
        """
        if not self.metrics_history:
            return {
                'status': 'unknown',
                'message': 'No metrics recorded yet',
            }
        
        latest_metrics = self.metrics_history[-1]
        alerts = []
        
        # Check against baselines
        for metric, threshold in self.alert_thresholds.items():
            if metric in self.baseline_metrics and metric in latest_metrics:
                baseline = self.baseline_metrics[metric]
                current = latest_metrics[metric]
                degradation = baseline - current
                degradation_pct = degradation / baseline if baseline > 0 else 0
                
                if degradation_pct > threshold:
                    alerts.append({
                        'metric': metric,
                        'baseline': baseline,
                        'current': current,
                        'degradation': degradation,
                        'degradation_pct': degradation_pct,
                        'threshold': threshold,
                        'severity': 'critical' if degradation_pct > threshold * 2 else 'warning',
                    })
        
        # Determine overall status
        if not alerts:
            status = 'healthy'
        elif any(a['severity'] == 'critical' for a in alerts):
            status = 'critical'
        else:
            status = 'warning'
        
        health = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'latest_metrics': latest_metrics,
            'baseline_metrics': self.baseline_metrics,
            'alerts': alerts,
            'metrics_count': len(self.metrics_history),
        }
        
        if alerts:
            self.logger.warning(
                "Model health check detected issues",
                status=status,
                num_alerts=len(alerts),
            )
        
        return health
    
    def get_metrics_summary(
        self,
        window_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get summary of metrics over time.
        
        Args:
            window_size: Number of recent records to include (None = all)
        
        Returns:
            DataFrame with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.metrics_history)
        
        if window_size:
            df = df.tail(window_size)
        
        return df
    
    def plot_metrics_over_time(
        self,
        metrics: Optional[List[str]] = None,
        window_size: Optional[int] = None,
    ) -> None:
        """
        Plot metrics over time.
        
        Args:
            metrics: Metrics to plot (None = all)
            window_size: Number of recent records
        """
        import matplotlib.pyplot as plt
        
        df = self.get_metrics_summary(window_size)
        
        if df.empty:
            print("No metrics to plot")
            return
        
        if metrics is None:
            metrics = ['auc', 'precision', 'recall', 'f1']
        
        # Filter to available metrics
        metrics = [m for m in metrics if m in df.columns]
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            ax.plot(df.index, df[metric], marker='o', label='Current')
            
            # Add baseline if available
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                ax.axhline(baseline, color='green', linestyle='--', label='Baseline')
                
                # Add alert threshold
                threshold = self.alert_thresholds.get(metric, 0)
                alert_line = baseline * (1 - threshold)
                ax.axhline(alert_line, color='red', linestyle='--', label='Alert Threshold')
            
            ax.set_xlabel('Batch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Over Time')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_report(
        self,
        filepath: str,
        format: str = 'html',
    ) -> None:
        """
        Export monitoring report.
        
        Args:
            filepath: Output file path
            format: Report format ('html', 'json', 'excel')
        """
        health = self.check_health()
        summary = self.get_metrics_summary()
        
        if format == 'json':
            report = {
                'health': health,
                'summary': summary.to_dict('records'),
            }
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == 'excel':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='Metrics History', index=False)
                pd.DataFrame(health['alerts']).to_excel(
                    writer, sheet_name='Alerts', index=False
                )
        
        elif format == 'html':
            html = f"""
            <html>
            <head><title>Model Performance Report</title></head>
            <body>
                <h1>Model Performance Monitoring Report</h1>
                <h2>Health Status: {health['status'].upper()}</h2>
                <p>Generated: {health['timestamp']}</p>
                
                <h3>Latest Metrics</h3>
                <table border="1">
                    {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' 
                             for k, v in health['latest_metrics'].items())}
                </table>
                
                <h3>Alerts</h3>
                {pd.DataFrame(health['alerts']).to_html() if health['alerts'] else '<p>No alerts</p>'}
                
                <h3>Metrics History</h3>
                {summary.to_html()}
            </body>
            </html>
            """
            with open(filepath, 'w') as f:
                f.write(html)
        
        self.logger.info(f"Performance report exported to {filepath}")
