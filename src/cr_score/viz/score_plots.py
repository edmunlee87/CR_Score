"""
Score visualization for scorecard analysis.

Creates interactive plots for score distributions, performance metrics,
and model diagnostics using Plotly.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    
    # ========================================================================
    # Temporal Stability Visualization Methods
    # ========================================================================
    
    def plot_temporal_score_distribution(
        self,
        df: pd.DataFrame,
        score_col: str,
        snapshot_col: str,
        snapshot_values: Optional[List] = None,
        target_col: Optional[str] = None,
        segment_col: Optional[str] = None,
        segment_values: Optional[List] = None,
        n_bins: int = 20,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot score distribution across multiple snapshots.
        
        Args:
            df: DataFrame with score and snapshot columns
            score_col: Name of score column
            snapshot_col: Name of snapshot/time column
            snapshot_values: List of snapshots to include (None = all)
            target_col: Optional target column for colored histograms
            segment_col: Optional segmentation column
            segment_values: List of segment values to filter (None = all)
            n_bins: Number of histogram bins
            title: Plot title
            
        Returns:
            Plotly figure with score distributions over time
            
        Example:
            >>> fig = visualizer.plot_temporal_score_distribution(
            ...     df, "credit_score", "month_end",
            ...     snapshot_values=["2024-01", "2024-06", "2024-12"]
            ... )
        """
        # Filter data
        if snapshot_values:
            df = df[df[snapshot_col].isin(snapshot_values)]
        if segment_col and segment_values:
            df = df[df[segment_col].isin(segment_values)]
        
        snapshots = sorted(df[snapshot_col].dropna().unique())
        colors = plotly.colors.qualitative.Set3[:len(snapshots)]
        
        fig = go.Figure()
        
        # Plot distribution for each snapshot
        for i, snapshot in enumerate(snapshots):
            df_snap = df[df[snapshot_col] == snapshot]
            
            if target_col:
                # Separate by target class
                scores_good = df_snap[df_snap[target_col] == 0][score_col].dropna()
                scores_bad = df_snap[df_snap[target_col] == 1][score_col].dropna()
                
                if len(scores_good) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores_good,
                            nbinsx=n_bins,
                            name=f"Good {snapshot}",
                            marker_color=colors[i],
                            opacity=0.6,
                            legendgroup=f"good_{snapshot}",
                        )
                    )
                
                if len(scores_bad) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores_bad,
                            nbinsx=n_bins,
                            name=f"Bad {snapshot}",
                            marker_color=colors[i],
                            opacity=0.8,
                            legendgroup=f"bad_{snapshot}",
                        )
                    )
            else:
                scores = df_snap[score_col].dropna()
                if len(scores) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=scores,
                            nbinsx=n_bins,
                            name=str(snapshot),
                            marker_color=colors[i],
                            opacity=0.6,
                        )
                    )
        
        plot_title = title or "Score Distribution by Snapshot"
        fig.update_layout(
            title_text=plot_title,
            xaxis_title="Credit Score",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white",
            height=500,
        )
        
        return fig
    
    def plot_temporal_ks_comparison(
        self,
        df: pd.DataFrame,
        score_col: str,
        target_col: str,
        snapshot_col: str,
        snapshot_values: Optional[List] = None,
        segment_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot KS curve comparison across snapshots.
        
        Args:
            df: DataFrame with score, target, and snapshot columns
            score_col: Name of score column
            target_col: Name of target column
            snapshot_col: Name of snapshot/time column
            snapshot_values: List of snapshots to include (None = all)
            segment_col: Optional segmentation column
            title: Plot title
            
        Returns:
            Plotly figure with KS curves for each snapshot
            
        Example:
            >>> fig = visualizer.plot_temporal_ks_comparison(
            ...     df, "credit_score", "default", "month_end"
            ... )
        """
        # Filter data
        if snapshot_values:
            df = df[df[snapshot_col].isin(snapshot_values)]
        
        snapshots = sorted(df[snapshot_col].dropna().unique())
        colors = plotly.colors.qualitative.Set3[:len(snapshots)]
        
        fig = go.Figure()
        
        for i, snapshot in enumerate(snapshots):
            df_snap = df[df[snapshot_col] == snapshot].copy()
            scores = df_snap[score_col].dropna().values
            targets = df_snap[target_col].dropna().values
            
            if len(scores) == 0 or len(targets) == 0:
                continue
            
            # Sort by score
            sorted_indices = np.argsort(scores)
            y_sorted = targets[sorted_indices]
            
            n_total = len(targets)
            n_bad = targets.sum()
            n_good = n_total - n_bad
            
            if n_bad == 0 or n_good == 0:
                continue
            
            cum_bad = np.cumsum(y_sorted) / n_bad
            cum_good = np.cumsum(1 - y_sorted) / n_good
            percentiles = np.arange(n_total) / n_total * 100
            
            # KS statistic
            ks_values = cum_bad - cum_good
            ks_max = np.max(ks_values)
            
            # Plot cumulative distributions
            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=cum_bad,
                    mode='lines',
                    name=f'Cum Bad % {snapshot}',
                    line=dict(color=colors[i], width=2, dash='solid'),
                    legendgroup=f"bad_{snapshot}",
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=percentiles,
                    y=cum_good,
                    mode='lines',
                    name=f'Cum Good % {snapshot}',
                    line=dict(color=colors[i], width=2, dash='dot'),
                    legendgroup=f"good_{snapshot}",
                )
            )
            
            # Add KS annotation
            ks_idx = np.argmax(ks_values)
            fig.add_annotation(
                x=percentiles[ks_idx],
                y=(cum_good[ks_idx] + cum_bad[ks_idx]) / 2,
                text=f"KS={ks_max:.3f}",
                showarrow=True,
                arrowhead=2,
                bgcolor="white",
                font=dict(size=9),
            )
        
        plot_title = title or "KS Curve Comparison by Snapshot"
        fig.update_layout(
            title_text=plot_title,
            xaxis_title="Population %",
            yaxis_title="Cumulative %",
            template="plotly_white",
            height=500,
        )
        
        return fig
    
    def plot_temporal_stability_metrics(
        self,
        df: pd.DataFrame,
        score_col: str,
        target_col: str,
        snapshot_col: str,
        snapshot_values: Optional[List] = None,
        approval_threshold: Optional[float] = None,
        segment_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot stability metrics (approval rate, bad rate, capture rate) over time.
        
        Args:
            df: DataFrame with score, target, and snapshot columns
            score_col: Name of score column
            target_col: Name of target column
            snapshot_col: Name of snapshot/time column
            snapshot_values: List of snapshots to include (None = all)
            approval_threshold: Score threshold for approval rate calculation
            segment_col: Optional segmentation column
            title: Plot title
            
        Returns:
            Plotly figure with stability metrics over time
            
        Example:
            >>> fig = visualizer.plot_temporal_stability_metrics(
            ...     df, "credit_score", "default", "month_end",
            ...     approval_threshold=600
            ... )
        """
        # Filter data
        if snapshot_values:
            df = df[df[snapshot_col].isin(snapshot_values)]
        
        snapshots = sorted(df[snapshot_col].dropna().unique())
        
        metrics = {
            'snapshot': [],
            'approval_rate': [],
            'bad_rate': [],
            'capture_rate_top_decile': [],
            'capture_rate_top_quintile': [],
        }
        
        for snapshot in snapshots:
            df_snap = df[df[snapshot_col] == snapshot].copy()
            
            if len(df_snap) == 0:
                continue
            
            # Bad rate
            bad_rate = df_snap[target_col].mean() if target_col in df_snap.columns else None
            
            # Approval rate (if threshold provided)
            if approval_threshold:
                approval_rate = (df_snap[score_col] >= approval_threshold).mean()
            else:
                approval_rate = None
            
            # Capture rate (top decile/quintile)
            if target_col in df_snap.columns:
                df_snap_sorted = df_snap.sort_values(score_col, ascending=False)
                n_top_decile = max(1, len(df_snap) // 10)
                n_top_quintile = max(1, len(df_snap) // 5)
                
                total_bad = df_snap[target_col].sum()
                if total_bad > 0:
                    capture_top_decile = df_snap_sorted.head(n_top_decile)[target_col].sum() / total_bad
                    capture_top_quintile = df_snap_sorted.head(n_top_quintile)[target_col].sum() / total_bad
                else:
                    capture_top_decile = capture_top_quintile = None
            else:
                capture_top_decile = capture_top_quintile = None
            
            metrics['snapshot'].append(str(snapshot))
            metrics['approval_rate'].append(approval_rate)
            metrics['bad_rate'].append(bad_rate)
            metrics['capture_rate_top_decile'].append(capture_top_decile)
            metrics['capture_rate_top_quintile'].append(capture_top_quintile)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Approval Rate", "Bad Rate", "Capture Rate (Top Decile)", "Capture Rate (Top Quintile)"),
            vertical_spacing=0.15,
        )
        
        # Plot each metric
        if any(m is not None for m in metrics['approval_rate']):
            fig.add_trace(
                go.Scatter(
                    x=metrics['snapshot'],
                    y=metrics['approval_rate'],
                    mode='lines+markers',
                    name='Approval Rate',
                    line=dict(color='blue', width=2),
                ),
                row=1, col=1,
            )
        
        if any(m is not None for m in metrics['bad_rate']):
            fig.add_trace(
                go.Scatter(
                    x=metrics['snapshot'],
                    y=metrics['bad_rate'],
                    mode='lines+markers',
                    name='Bad Rate',
                    line=dict(color='red', width=2),
                ),
                row=1, col=2,
            )
        
        if any(m is not None for m in metrics['capture_rate_top_decile']):
            fig.add_trace(
                go.Scatter(
                    x=metrics['snapshot'],
                    y=metrics['capture_rate_top_decile'],
                    mode='lines+markers',
                    name='Capture (Top 10%)',
                    line=dict(color='green', width=2),
                ),
                row=2, col=1,
            )
        
        if any(m is not None for m in metrics['capture_rate_top_quintile']):
            fig.add_trace(
                go.Scatter(
                    x=metrics['snapshot'],
                    y=metrics['capture_rate_top_quintile'],
                    mode='lines+markers',
                    name='Capture (Top 20%)',
                    line=dict(color='orange', width=2),
                ),
                row=2, col=2,
            )
        
        plot_title = title or "Score Stability Metrics by Snapshot"
        fig.update_layout(
            title_text=plot_title,
            height=700,
            template="plotly_white",
            showlegend=False,
        )
        
        fig.update_xaxes(title_text="Snapshot", row=2, col=1)
        fig.update_xaxes(title_text="Snapshot", row=2, col=2)
        
        return fig
    
    def _export_figure_with_metadata(
        self,
        fig: go.Figure,
        path: str,
        format: str = "html",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export figure with embedded metadata.
        
        Args:
            fig: Plotly figure
            path: Export path
            format: Export format ("html" or "png")
            metadata: Dictionary of metadata to embed
        """
        if metadata:
            metadata_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in metadata.items()])
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=-0.1,
                xanchor="left", yanchor="top",
                text=metadata_text,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
            
            fig.update_layout(margin=dict(b=100))
        
        if format == "html":
            fig.write_html(path)
        elif format == "png":
            fig.write_image(path, width=1200, height=800, scale=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported figure to {path}", metadata=metadata)
