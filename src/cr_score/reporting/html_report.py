"""
HTML report generation for scorecard documentation.

Creates comprehensive HTML reports with interactive visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from jinja2 import Template

from cr_score.core.logging import get_audit_logger
from cr_score.viz import BinningVisualizer, ScoreVisualizer


class HTMLReportGenerator:
    """
    Generate comprehensive HTML reports for scorecards.

    Creates professional HTML documentation with:
    - Executive summary
    - Model performance metrics
    - Feature analysis
    - Interactive visualizations
    - Model scorecard table

    Example:
        >>> generator = HTMLReportGenerator()
        >>> generator.generate_scorecard_report(
        ...     pipeline=pipeline,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ...     output_path="scorecard_report.html"
        ... )
    """

    def __init__(self) -> None:
        """Initialize HTML report generator."""
        self.logger = get_audit_logger()
        self.bin_visualizer = BinningVisualizer()
        self.score_visualizer = ScoreVisualizer()

    def generate_scorecard_report(
        self,
        pipeline: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_path: str,
        title: str = "Credit Scorecard Report",
        author: str = "CR_Score Platform",
    ) -> str:
        """
        Generate comprehensive scorecard report.

        Args:
            pipeline: Fitted ScorecardPipeline
            X_test: Test features
            y_test: Test target
            output_path: Output HTML file path
            title: Report title
            author: Report author

        Returns:
            Path to generated HTML file

        Example:
            >>> report_path = generator.generate_scorecard_report(
            ...     pipeline=pipeline,
            ...     X_test=X_test,
            ...     y_test=y_test,
            ...     output_path="report.html"
            ... )
        """
        self.logger.info("Generating scorecard report", output=output_path)

        # Get predictions
        scores = pipeline.predict(X_test)
        probas = pipeline.predict_proba(X_test)

        # Get pipeline summary
        summary = pipeline.get_summary()

        # Get performance metrics
        from sklearn.metrics import roc_auc_score

        metrics = pipeline.model_.get_performance_metrics(y_test, probas)

        # Create visualizations
        plots = self._create_plots(y_test, probas, scores, summary)

        # Generate HTML
        html = self._build_html(
            title=title,
            author=author,
            summary=summary,
            metrics=metrics,
            plots=plots,
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding="utf-8")

        self.logger.info(f"Report generated successfully: {output_path}")

        return str(output_file)

    def _create_plots(
        self,
        y_true: np.ndarray,
        probas: np.ndarray,
        scores: np.ndarray,
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        """Create visualization plots as HTML."""
        plots = {}

        # ROC Curve
        fig = self.score_visualizer.plot_roc_curve(y_true, probas)
        plots["roc_curve"] = fig.to_html(include_plotlyjs=False, div_id="roc_curve")

        # Score Distribution
        fig = self.score_visualizer.plot_score_distribution(scores, y_true)
        plots["score_dist"] = fig.to_html(include_plotlyjs=False, div_id="score_dist")

        # Score Bands
        fig = self.score_visualizer.plot_score_bands(scores, y_true, n_bands=10)
        plots["score_bands"] = fig.to_html(include_plotlyjs=False, div_id="score_bands")

        # Calibration Curve
        fig = self.score_visualizer.plot_calibration_curve(y_true, probas)
        plots["calibration"] = fig.to_html(include_plotlyjs=False, div_id="calibration")

        # KS Statistic
        fig = self.score_visualizer.plot_ks_statistic(y_true, probas)
        plots["ks_curve"] = fig.to_html(include_plotlyjs=False, div_id="ks_curve")

        return plots

    def _build_html(
        self,
        title: str,
        author: str,
        summary: Dict[str, Any],
        metrics: Dict[str, Any],
        plots: Dict[str, str],
    ) -> str:
        """Build HTML report from template."""
        template = Template(HTML_TEMPLATE)

        html = template.render(
            title=title,
            author=author,
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            metrics=metrics,
            plots=plots,
        )

        return html


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .meta {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        h2 {
            color: #667eea;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        h3 {
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .summary-card h3 {
            color: #333;
            font-size: 1em;
            margin: 0 0 10px 0;
            font-weight: 600;
        }
        
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }
        
        .metrics-table tr:hover {
            background-color: #f5f7fa;
        }
        
        .feature-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        
        .feature-tag {
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .plot-container {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert-info {
            background-color: #e3f2fd;
            border-color: #2196f3;
            color: #1976d2;
        }
        
        .alert-success {
            background-color: #e8f5e9;
            border-color: #4caf50;
            color: #388e3c;
        }
        
        footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <div class="meta">
                <p>Generated by {{ author }}</p>
                <p>{{ generated_date }}</p>
            </div>
        </header>
        
        <section>
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Features Selected</h3>
                    <div class="value">{{ summary.n_features }}</div>
                </div>
                <div class="summary-card">
                    <h3>AUC Score</h3>
                    <div class="value">{{ "%.3f"|format(metrics.auc) }}</div>
                </div>
                <div class="summary-card">
                    <h3>Gini Coefficient</h3>
                    <div class="value">{{ "%.3f"|format(metrics.gini) }}</div>
                </div>
                <div class="summary-card">
                    <h3>KS Statistic</h3>
                    <div class="value">{{ "%.3f"|format(metrics.ks) }}</div>
                </div>
            </div>
            
            {% if summary.get('feature_selection_method') %}
            <div class="alert alert-info">
                <strong>Feature Selection:</strong> Used {{ summary.feature_selection_method }} method. 
                Reduced from {{ summary.features_before_selection }} to {{ summary.features_after_selection }} features.
            </div>
            {% endif %}
        </section>
        
        <section>
            <h2>Model Performance Metrics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>AUC (Area Under ROC)</td>
                        <td>{{ "%.4f"|format(metrics.auc) }}</td>
                        <td>{% if metrics.auc >= 0.8 %}Excellent{% elif metrics.auc >= 0.7 %}Good{% else %}Fair{% endif %}</td>
                    </tr>
                    <tr>
                        <td>Gini Coefficient</td>
                        <td>{{ "%.4f"|format(metrics.gini) }}</td>
                        <td>Model discrimination power</td>
                    </tr>
                    <tr>
                        <td>KS Statistic</td>
                        <td>{{ "%.4f"|format(metrics.ks) }}</td>
                        <td>Maximum separation between good/bad</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{{ "%.4f"|format(metrics.accuracy) }}</td>
                        <td>Overall prediction accuracy</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{{ "%.4f"|format(metrics.precision) }}</td>
                        <td>Positive prediction accuracy</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{{ "%.4f"|format(metrics.recall) }}</td>
                        <td>True positive rate</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td>{{ "%.4f"|format(metrics.f1_score) }}</td>
                        <td>Harmonic mean of precision and recall</td>
                    </tr>
                </tbody>
            </table>
        </section>
        
        <section>
            <h2>Selected Features</h2>
            <p>The following {{ summary.n_features }} features were selected for the scorecard:</p>
            <div class="feature-list">
                {% for feature in summary.selected_features %}
                <div class="feature-tag">{{ feature }}</div>
                {% endfor %}
            </div>
        </section>
        
        <section>
            <h2>Performance Visualizations</h2>
            
            <h3>ROC Curve</h3>
            <div class="plot-container">
                {{ plots.roc_curve|safe }}
            </div>
            
            <h3>Score Distribution</h3>
            <div class="plot-container">
                {{ plots.score_dist|safe }}
            </div>
            
            <h3>Score Band Analysis</h3>
            <div class="plot-container">
                {{ plots.score_bands|safe }}
            </div>
            
            <h3>Calibration Curve</h3>
            <div class="plot-container">
                {{ plots.calibration|safe }}
            </div>
            
            <h3>Kolmogorov-Smirnov Statistic</h3>
            <div class="plot-container">
                {{ plots.ks_curve|safe }}
            </div>
        </section>
        
        <section>
            <h2>Model Configuration</h2>
            <table class="metrics-table">
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Points to Double Odds (PDO)</td>
                    <td>{{ summary.pdo_params.pdo }}</td>
                </tr>
                <tr>
                    <td>Base Score</td>
                    <td>{{ summary.pdo_params.base_score }}</td>
                </tr>
                <tr>
                    <td>Base Odds</td>
                    <td>{{ summary.pdo_params.base_odds }}</td>
                </tr>
                <tr>
                    <td>Calibrated</td>
                    <td>{{ "Yes" if summary.calibrated else "No" }}</td>
                </tr>
            </table>
        </section>
        
        <footer>
            <p>Generated by CR_Score Platform - Enterprise Scorecard Development</p>
            <p>For more information, visit the documentation</p>
        </footer>
    </div>
</body>
</html>
"""
