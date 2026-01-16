"""
Score visualization for scorecard analysis.

Creates interactive plots for score distributions, performance metrics,
and model diagnostics using Plotly.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

from cr_score.core.logging import get_audit_logger


class ScoreVisualizer:
    """
    Create visualizations for scorecard analysis.

    Generates interactive Plotly charts for score distributions,
    ROC curves, calibration plots, and performance metrics.

    Example:
        >>> visualizer = ScoreVisualizer()
        >>> fig = visualizer.plot_score_distribution(scores, y_true)
        >>> fig.show()
    """

    def __init__(self) -> None:
        """Initialize score visualizer."""
        self.logger = get_audit_logger()

    def plot_score_distribution(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 20,
    ) -> go.Figure:
        """
        Plot score distribution by target class.

        Args:
            scores: Credit scores
            y_true: True labels
            n_bins: Number of bins for histogram

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_score_distribution(scores, y_test)
            >>> fig.write_html("score_distribution.html")
        """
        fig = go.Figure()

        # Non-defaults
        scores_good = scores[y_true == 0]
        fig.add_trace(
            go.Histogram(
                x=scores_good,
                nbinsx=n_bins,
                name="Good (0)",
                marker_color="green",
                opacity=0.6,
            )
        )

        # Defaults
        scores_bad = scores[y_true == 1]
        fig.add_trace(
            go.Histogram(
                x=scores_bad,
                nbinsx=n_bins,
                name="Bad (1)",
                marker_color="red",
                opacity=0.6,
            )
        )

        fig.update_layout(
            title="Score Distribution by Target Class",
            xaxis_title="Credit Score",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white",
            height=500,
        )

        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
    ) -> go.Figure:
        """
        Plot ROC curve with AUC.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title

        Returns:
            Plotly figure

        Example:
            >>> probas = model.predict_proba(X_test)[:, 1]
            >>> fig = visualizer.plot_roc_curve(y_test, probas)
            >>> fig.show()
        """
        from sklearn.metrics import auc

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Calculate KS statistic
        ks = np.max(tpr - fpr)
        ks_idx = np.argmax(tpr - fpr)

        fig = go.Figure()

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC={roc_auc:.3f})",
                line=dict(color="blue", width=2),
            )
        )

        # Diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )

        # KS point
        fig.add_trace(
            go.Scatter(
                x=[fpr[ks_idx]],
                y=[tpr[ks_idx]],
                mode="markers",
                name=f"KS={ks:.3f}",
                marker=dict(color="red", size=10),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=500,
            width=600,
        )

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> go.Figure:
        """
        Plot calibration curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_calibration_curve(y_test, probas)
            >>> fig.show()
        """
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate actual vs predicted by bin
        bin_sums = np.bincount(bin_indices, weights=y_pred_proba, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
        bin_total = np.bincount(bin_indices, minlength=n_bins)

        # Avoid division by zero
        nonzero = bin_total > 0
        fraction_positives = np.zeros(n_bins)
        mean_predicted_value = np.zeros(n_bins)

        fraction_positives[nonzero] = bin_true[nonzero] / bin_total[nonzero]
        mean_predicted_value[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

        fig = go.Figure()

        # Calibration curve
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_positives,
                mode="lines+markers",
                name="Calibration",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )

        fig.update_layout(
            title="Calibration Curve",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Actual Fraction of Positives",
            template="plotly_white",
            height=500,
            width=600,
        )

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> go.Figure:
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Plotly figure

        Example:
            >>> predictions = (probas >= 0.5).astype(int)
            >>> fig = visualizer.plot_confusion_matrix(y_test, predictions)
            >>> fig.show()
        """
        cm = confusion_matrix(y_true, y_pred)

        # Create labels with counts and percentages
        labels = []
        for i in range(2):
            row = []
            for j in range(2):
                count = cm[i, j]
                total = cm.sum()
                pct = 100 * count / total
                row.append(f"{count}<br>({pct:.1f}%)")
            labels.append(row)

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Good (0)", "Predicted Bad (1)"],
                y=["Actual Good (0)", "Actual Bad (1)"],
                text=labels,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True,
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            template="plotly_white",
            height=500,
            width=600,
        )

        return fig

    def plot_score_bands(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_bands: int = 10,
    ) -> go.Figure:
        """
        Create score band analysis.

        Args:
            scores: Credit scores
            y_true: True labels
            n_bands: Number of score bands

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_score_bands(scores, y_test)
            >>> fig.show()
        """
        # Create score bands
        bands = pd.qcut(scores, q=n_bands, duplicates="drop")

        df = pd.DataFrame({
            "score": scores,
            "target": y_true,
            "band": bands,
        })

        # Calculate statistics by band
        stats = df.groupby("band", observed=True).agg({
            "target": ["count", "sum", "mean"],
        })

        stats.columns = ["count", "events", "event_rate"]
        stats = stats.reset_index()
        stats["band_label"] = stats["band"].astype(str)

        # Create subplot
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar for volume
        fig.add_trace(
            go.Bar(
                x=stats["band_label"],
                y=stats["count"],
                name="Volume",
                marker_color="lightblue",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        # Add line for event rate
        fig.add_trace(
            go.Scatter(
                x=stats["band_label"],
                y=stats["event_rate"],
                name="Default Rate",
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=10),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Score Band Analysis",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Score Band", tickangle=-45)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Default Rate", secondary_y=True)

        return fig

    def plot_ks_statistic(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> go.Figure:
        """
        Plot KS statistic curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_ks_statistic(y_test, probas)
            >>> fig.show()
        """
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred_proba)
        y_true_sorted = y_true[sorted_indices]

        # Calculate cumulative distributions
        n_total = len(y_true)
        n_bad = y_true.sum()
        n_good = n_total - n_bad

        cum_bad = np.cumsum(y_true_sorted) / n_bad
        cum_good = np.cumsum(1 - y_true_sorted) / n_good

        # KS statistic
        ks_values = cum_bad - cum_good
        ks_max = np.max(ks_values)
        ks_idx = np.argmax(ks_values)

        percentiles = np.arange(len(y_true)) / len(y_true) * 100

        fig = go.Figure()

        # Cumulative bad rate
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=cum_bad,
                mode="lines",
                name="Cumulative Bad %",
                line=dict(color="red", width=2),
            )
        )

        # Cumulative good rate
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=cum_good,
                mode="lines",
                name="Cumulative Good %",
                line=dict(color="green", width=2),
            )
        )

        # KS line
        fig.add_shape(
            type="line",
            x0=percentiles[ks_idx],
            y0=cum_good[ks_idx],
            x1=percentiles[ks_idx],
            y1=cum_bad[ks_idx],
            line=dict(color="blue", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=percentiles[ks_idx],
            y=(cum_good[ks_idx] + cum_bad[ks_idx]) / 2,
            text=f"KS={ks_max:.3f}",
            showarrow=True,
            arrowhead=2,
            bgcolor="white",
        )

        fig.update_layout(
            title="Kolmogorov-Smirnov (KS) Statistic",
            xaxis_title="Population %",
            yaxis_title="Cumulative %",
            template="plotly_white",
            height=500,
        )

        return fig

    def create_model_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """
        Create comprehensive model performance report.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            scores: Credit scores (optional)

        Returns:
            Plotly figure with multiple subplots

        Example:
            >>> fig = visualizer.create_model_report(y_test, probas, scores)
            >>> fig.write_html("model_report.html")
        """
        from sklearn.metrics import auc

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ks = np.max(tpr - fpr)

        # Create subplots
        if scores is not None:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    f"ROC Curve (AUC={roc_auc:.3f})",
                    "Calibration Curve",
                    "Score Distribution",
                    f"KS Statistic (KS={ks:.3f})",
                ),
            )
        else:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    f"ROC Curve (AUC={roc_auc:.3f})",
                    "Calibration Curve",
                ),
            )

        # ROC Curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="blue", width=2)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="gray", dash="dash")),
            row=1,
            col=1,
        )

        # Calibration curve
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(y_pred_proba, bins) - 1, 0, n_bins - 1)

        bin_sums = np.bincount(bin_indices, weights=y_pred_proba, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
        bin_total = np.bincount(bin_indices, minlength=n_bins)

        nonzero = bin_total > 0
        fraction_positives = np.zeros(n_bins)
        mean_predicted = np.zeros(n_bins)
        fraction_positives[nonzero] = bin_true[nonzero] / bin_total[nonzero]
        mean_predicted[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

        fig.add_trace(
            go.Scatter(x=mean_predicted, y=fraction_positives, mode="lines+markers", name="Calibration"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color="gray", dash="dash")),
            row=1,
            col=2,
        )

        # Score distribution (if provided)
        if scores is not None:
            fig.add_trace(
                go.Histogram(x=scores[y_true == 0], name="Good", marker_color="green", opacity=0.6, nbinsx=20),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Histogram(x=scores[y_true == 1], name="Bad", marker_color="red", opacity=0.6, nbinsx=20),
                row=2,
                col=1,
            )

            # KS Curve
            sorted_indices = np.argsort(y_pred_proba)
            y_sorted = y_true[sorted_indices]
            n_bad = y_true.sum()
            n_good = len(y_true) - n_bad

            cum_bad = np.cumsum(y_sorted) / n_bad
            cum_good = np.cumsum(1 - y_sorted) / n_good
            percentiles = np.arange(len(y_true)) / len(y_true) * 100

            fig.add_trace(
                go.Scatter(x=percentiles, y=cum_bad, mode="lines", name="Bad %", line=dict(color="red")),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Scatter(x=percentiles, y=cum_good, mode="lines", name="Good %", line=dict(color="green")),
                row=2,
                col=2,
            )

        fig.update_layout(
            title_text="Model Performance Report",
            showlegend=True,
            height=800 if scores is not None else 400,
            template="plotly_white",
        )

        return fig
