"""
Observability dashboard for production scorecard monitoring.

Creates interactive dashboards for monitoring model performance, drift, and system health.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cr_score.core.logging import get_audit_logger


class ObservabilityDashboard:
    """
    Create observability dashboards for scorecard monitoring.
    
    Generates interactive HTML dashboards with real-time metrics.
    
    Example:
        >>> dashboard = ObservabilityDashboard()
        >>> dashboard.add_performance_metrics(performance_data)
        >>> dashboard.add_drift_metrics(drift_data)
        >>> dashboard.export("dashboard.html")
    """
    
    def __init__(self, title: str = "Scorecard Observability Dashboard"):
        """
        Initialize dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.logger = get_audit_logger()
        
        self.sections: List[Dict[str, Any]] = []
    
    def add_performance_section(
        self,
        metrics_df: pd.DataFrame,
        health_status: Dict[str, Any],
    ) -> None:
        """
        Add performance monitoring section.
        
        Args:
            metrics_df: Performance metrics over time
            health_status: Current health status
        """
        section = {
            'title': 'Performance Monitoring',
            'type': 'performance',
            'data': {
                'metrics': metrics_df.to_dict('records'),
                'health': health_status,
            }
        }
        
        self.sections.append(section)
    
    def add_drift_section(
        self,
        drift_report: Dict[str, Any],
    ) -> None:
        """
        Add drift monitoring section.
        
        Args:
            drift_report: Drift detection report
        """
        section = {
            'title': 'Data Drift Monitoring',
            'type': 'drift',
            'data': drift_report,
        }
        
        self.sections.append(section)
    
    def add_prediction_section(
        self,
        prediction_stats: pd.DataFrame,
    ) -> None:
        """
        Add prediction monitoring section.
        
        Args:
            prediction_stats: Prediction statistics
        """
        section = {
            'title': 'Prediction Monitoring',
            'type': 'prediction',
            'data': prediction_stats.to_dict('records'),
        }
        
        self.sections.append(section)
    
    def add_metrics_section(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Add system metrics section.
        
        Args:
            metrics: System metrics
        """
        section = {
            'title': 'System Metrics',
            'type': 'metrics',
            'data': metrics,
        }
        
        self.sections.append(section)
    
    def add_alerts_section(
        self,
        active_alerts: List[Dict[str, Any]],
        alert_summary: Dict[str, Any],
    ) -> None:
        """
        Add alerts section.
        
        Args:
            active_alerts: Active alerts
            alert_summary: Alert summary
        """
        section = {
            'title': 'Active Alerts',
            'type': 'alerts',
            'data': {
                'active': active_alerts,
                'summary': alert_summary,
            }
        }
        
        self.sections.append(section)
    
    def export(self, filepath: str) -> None:
        """
        Export dashboard to HTML.
        
        Args:
            filepath: Output file path
        """
        html = self._generate_html()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"Observability dashboard exported to {filepath}")
    
    def _generate_html(self) -> str:
        """Generate HTML dashboard."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .metric-card {{
            display: inline-block;
            background-color: #ecf0f1;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #27ae60;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .alert-critical {{
            background-color: #e74c3c;
            color: white;
        }}
        .alert-warning {{
            background-color: #f39c12;
            color: white;
        }}
        .alert-info {{
            background-color: #3498db;
            color: white;
        }}
        .status-healthy {{
            color: #27ae60;
        }}
        .status-warning {{
            color: #f39c12;
        }}
        .status-critical {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Generated: {timestamp}</p>
    </div>
    
    {self._render_sections()}
</body>
</html>
"""
        
        return html
    
    def _render_sections(self) -> str:
        """Render all dashboard sections."""
        html_sections = []
        
        for section in self.sections:
            if section['type'] == 'performance':
                html_sections.append(self._render_performance(section))
            elif section['type'] == 'drift':
                html_sections.append(self._render_drift(section))
            elif section['type'] == 'prediction':
                html_sections.append(self._render_prediction(section))
            elif section['type'] == 'metrics':
                html_sections.append(self._render_metrics(section))
            elif section['type'] == 'alerts':
                html_sections.append(self._render_alerts(section))
        
        return "\n".join(html_sections)
    
    def _render_performance(self, section: Dict[str, Any]) -> str:
        """Render performance section."""
        data = section['data']
        health = data['health']
        
        status_class = f"status-{health['status']}"
        
        html = f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        <h2 class="{status_class}">Status: {health['status'].upper()}</h2>
        
        <div>
            <div class="metric-card">
                <div class="metric-value">{health['latest_metrics'].get('auc', 'N/A'):.3f}</div>
                <div class="metric-label">Current AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{health['latest_metrics'].get('precision', 'N/A'):.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{health['latest_metrics'].get('recall', 'N/A'):.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{health['metrics_count']}</div>
                <div class="metric-label">Metrics Collected</div>
            </div>
        </div>
        
        {self._render_alerts_inline(health.get('alerts', []))}
    </div>
"""
        
        return html
    
    def _render_drift(self, section: Dict[str, Any]) -> str:
        """Render drift section."""
        data = section['data']
        summary = data['drift_summary']
        
        html = f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        <h2 class="status-{data['overall_status']}">Status: {data['overall_status'].upper()}</h2>
        
        <div>
            <div class="metric-card">
                <div class="metric-value" style="color:#e74c3c">{summary['critical']}</div>
                <div class="metric-label">Critical Drift</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#f39c12">{summary['warning']}</div>
                <div class="metric-label">Warning Drift</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#27ae60">{summary['stable']}</div>
                <div class="metric-label">Stable</div>
            </div>
        </div>
        
        {self._render_table(data['drift_results'])}
    </div>
"""
        
        return html
    
    def _render_prediction(self, section: Dict[str, Any]) -> str:
        """Render prediction section."""
        data = section['data']
        
        html = f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        {self._render_table(data)}
    </div>
"""
        
        return html
    
    def _render_metrics(self, section: Dict[str, Any]) -> str:
        """Render metrics section."""
        data = section['data']
        
        html = f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        <div>
            <div class="metric-card">
                <div class="metric-value">{data.get('uptime_seconds', 0):.0f}s</div>
                <div class="metric-label">Uptime</div>
            </div>
        </div>
        <pre>{str(data)}</pre>
    </div>
"""
        
        return html
    
    def _render_alerts(self, section: Dict[str, Any]) -> str:
        """Render alerts section."""
        data = section['data']
        
        html = f"""
    <div class="section">
        <div class="section-title">{section['title']}</div>
        {self._render_alerts_inline(data['active'])}
    </div>
"""
        
        return html
    
    def _render_alerts_inline(self, alerts: List[Dict[str, Any]]) -> str:
        """Render alerts list."""
        if not alerts:
            return "<p>No active alerts</p>"
        
        alert_html = []
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            alert_html.append(f"""
        <div class="alert {severity_class}">
            <strong>{alert['title']}</strong>
            <p>{alert.get('details', {})}</p>
        </div>
            """)
        
        return "\n".join(alert_html)
    
    def _render_table(self, data: List[Dict[str, Any]]) -> str:
        """Render data table."""
        if not data:
            return "<p>No data available</p>"
        
        df = pd.DataFrame(data)
        return df.to_html(classes='table', index=False)
