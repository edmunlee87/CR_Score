"""
Binning visualization for scorecard analysis.

Creates interactive plots for bin analysis using Plotly.
Supports temporal drift visualization and multi-snapshot comparisons.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots
from scipy import stats

from cr_score.core.logging import get_audit_logger


class BinningVisualizer:
    """
    Create visualizations for binning analysis.

    Generates interactive Plotly charts for bin distributions,
    event rates, WoE values, and IV contributions.

    Example:
        >>> visualizer = BinningVisualizer()
        >>> fig = visualizer.plot_binning_table(binning_table, title="Age Binning")
        >>> fig.show()
    """

    def __init__(self) -> None:
        """Initialize binning visualizer."""
        self.logger = get_audit_logger()

    def plot_binning_table(
        self,
        binning_table: pd.DataFrame,
        title: str = "Binning Analysis",
    ) -> go.Figure:
        """
        Create comprehensive binning visualization.

        Args:
            binning_table: Binning table with columns: Bin, Count, Event rate, WoE, IV
            title: Plot title

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_binning_table(encoder.get_woe_table())
            >>> fig.write_html("binning_plot.html")
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Distribution",
                "Event Rate",
                "Weight of Evidence (WoE)",
                "IV Contribution",
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "bar"}, {"type": "bar"}]],
        )

        bins = binning_table.iloc[:, 0].astype(str)

        # Plot 1: Distribution
        fig.add_trace(
            go.Bar(
                x=bins,
                y=binning_table["total_count"] if "total_count" in binning_table else binning_table.get("count", []),
                name="Count",
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

        # Plot 2: Event Rate
        fig.add_trace(
            go.Scatter(
                x=bins,
                y=binning_table["event_rate"],
                mode="lines+markers",
                name="Event Rate",
                line=dict(color="red", width=2),
                marker=dict(size=8),
            ),
            row=1,
            col=2,
        )

        # Plot 3: WoE
        fig.add_trace(
            go.Bar(
                x=bins,
                y=binning_table["woe"] if "woe" in binning_table else [0] * len(bins),
                name="WoE",
                marker_color="green",
            ),
            row=2,
            col=1,
        )

        # Plot 4: IV Contribution
        fig.add_trace(
            go.Bar(
                x=bins,
                y=binning_table["iv_contribution"] if "iv_contribution" in binning_table else [0] * len(bins),
                name="IV",
                marker_color="purple",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Bin", row=1, col=1)
        fig.update_xaxes(title_text="Bin", row=1, col=2)
        fig.update_xaxes(title_text="Bin", row=2, col=1)
        fig.update_xaxes(title_text="Bin", row=2, col=2)

        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Event Rate", row=1, col=2)
        fig.update_yaxes(title_text="WoE", row=2, col=1)
        fig.update_yaxes(title_text="IV Contribution", row=2, col=2)

        return fig

    def plot_iv_summary(
        self,
        iv_summary: pd.DataFrame,
        top_n: int = 15,
    ) -> go.Figure:
        """
        Plot Information Value summary for all features.

        Args:
            iv_summary: DataFrame with columns: feature, iv
            top_n: Number of top features to show

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_iv_summary(encoder.get_iv_summary())
            >>> fig.show()
        """
        df_plot = iv_summary.head(top_n).sort_values("iv")

        fig = go.Figure()

        # Color coding by IV strength
        colors = []
        for iv in df_plot["iv"]:
            if iv < 0.02:
                colors.append("lightgray")
            elif iv < 0.1:
                colors.append("yellow")
            elif iv < 0.3:
                colors.append("orange")
            elif iv < 0.5:
                colors.append("green")
            else:
                colors.append("red")

        fig.add_trace(
            go.Bar(
                x=df_plot["iv"],
                y=df_plot["feature"],
                orientation="h",
                marker_color=colors,
                text=df_plot["iv"].round(3),
                textposition="outside",
            )
        )

        # Add reference lines
        fig.add_vline(x=0.02, line_dash="dash", line_color="gray", annotation_text="Weak")
        fig.add_vline(x=0.1, line_dash="dash", line_color="orange", annotation_text="Medium")
        fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Strong")

        fig.update_layout(
            title="Information Value (IV) by Feature",
            xaxis_title="Information Value",
            yaxis_title="Feature",
            height=max(400, len(df_plot) * 30),
            template="plotly_white",
            showlegend=False,
        )

        return fig

    def plot_feature_comparison(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
    ) -> go.Figure:
        """
        Create feature vs target comparison plot.

        Args:
            df: DataFrame
            feature_col: Feature column (can be binned or continuous)
            target_col: Target column

        Returns:
            Plotly figure

        Example:
            >>> fig = visualizer.plot_feature_comparison(df, "age_bin", "default")
            >>> fig.show()
        """
        # Calculate statistics by feature value
        stats = df.groupby(feature_col, dropna=False)[target_col].agg(
            count="count",
            events="sum",
        )

        stats["event_rate"] = stats["events"] / stats["count"]
        stats = stats.reset_index()

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar chart for counts
        fig.add_trace(
            go.Bar(
                x=stats[feature_col].astype(str),
                y=stats["count"],
                name="Count",
                marker_color="lightblue",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        # Add line chart for event rate
        fig.add_trace(
            go.Scatter(
                x=stats[feature_col].astype(str),
                y=stats["event_rate"],
                name="Event Rate",
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=10),
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title=f"Feature Analysis: {feature_col}",
            template="plotly_white",
            height=500,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text=feature_col)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Event Rate", secondary_y=True)

        return fig
    
    # ========================================================================
    # Temporal Drift Visualization Methods
    # ========================================================================
    
    def plot_temporal_bin_drift(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        snapshot_col: str,
        snapshot_values: Optional[List] = None,
        baseline_snapshot: Optional[str] = None,
        segment_col: Optional[str] = None,
        segment_values: Optional[List] = None,
        max_bins_display: int = 20,
        show_confidence_bands: bool = False,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot bin-level temporal drift across snapshots.
        
        Shows event rate and population % per bin across multiple snapshots.
        
        Args:
            df: DataFrame with feature, target, and snapshot columns
            feature_col: Name of binned feature column
            target_col: Name of target column
            snapshot_col: Name of snapshot/time column
            snapshot_values: List of snapshot values to include (None = all)
            baseline_snapshot: Reference snapshot for delta calculations
            segment_col: Optional segmentation column
            segment_values: List of segment values to filter (None = all)
            max_bins_display: Maximum number of bins to display
            show_confidence_bands: Whether to show confidence bands for event rate
            title: Plot title
            
        Returns:
            Plotly figure with temporal drift visualization
            
        Example:
            >>> fig = visualizer.plot_temporal_bin_drift(
            ...     df, "age_bin", "default", "month_end",
            ...     snapshot_values=["2024-01", "2024-06", "2024-12"],
            ...     baseline_snapshot="2024-01"
            ... )
        """
        # Filter by snapshot values if provided
        if snapshot_values:
            df = df[df[snapshot_col].isin(snapshot_values)]
        
        # Filter by segment if provided
        if segment_col and segment_values:
            df = df[df[segment_col].isin(segment_values)]
        
        # Get unique bins and snapshots
        bins = sorted(df[feature_col].dropna().unique())
        if len(bins) > max_bins_display:
            bins = bins[:max_bins_display]
            df = df[df[feature_col].isin(bins)]
        
        snapshots = sorted(df[snapshot_col].dropna().unique())
        if baseline_snapshot and baseline_snapshot not in snapshots:
            baseline_snapshot = snapshots[0]
        
        # Aggregate by bin and snapshot
        agg_data = []
        for snapshot in snapshots:
            df_snap = df[df[snapshot_col] == snapshot]
            for bin_val in bins:
                df_bin = df_snap[df_snap[feature_col] == bin_val]
                if len(df_bin) > 0:
                    count = len(df_bin)
                    events = df_bin[target_col].sum()
                    event_rate = events / count
                    pop_pct = count / len(df_snap)
                    
                    # Confidence interval for event rate (Wilson score)
                    if show_confidence_bands and count > 0:
                        z = 1.96  # 95% CI
                        denominator = 1 + (z**2 / count)
                        centre_adjusted = (event_rate + z**2 / (2 * count)) / denominator
                        margin_adjusted = z * np.sqrt((event_rate * (1 - event_rate) + z**2 / (4 * count)) / count) / denominator
                        ci_lower = max(0, centre_adjusted - margin_adjusted)
                        ci_upper = min(1, centre_adjusted + margin_adjusted)
                    else:
                        ci_lower = ci_upper = None
                    
                    agg_data.append({
                        'bin': bin_val,
                        'snapshot': snapshot,
                        'count': count,
                        'events': events,
                        'event_rate': event_rate,
                        'pop_pct': pop_pct,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'segment': df_bin[segment_col].iloc[0] if segment_col else None,
                    })
        
        df_agg = pd.DataFrame(agg_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Event Rate by Snapshot", "Population % by Snapshot"),
            vertical_spacing=0.1,
            shared_xaxes=True,
        )
        
        # Color palette for snapshots
        colors = plotly.colors.qualitative.Set3[:len(snapshots)]
        
        # Plot 1: Event rate by snapshot
        for i, snapshot in enumerate(snapshots):
            df_snap = df_agg[df_agg['snapshot'] == snapshot]
            df_snap = df_snap.sort_values('bin')
            
            fig.add_trace(
                go.Scatter(
                    x=df_snap['bin'].astype(str),
                    y=df_snap['event_rate'],
                    mode='lines+markers',
                    name=f'Event Rate {snapshot}',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                ),
                row=1, col=1,
            )
            
            # Add confidence bands if requested
            if show_confidence_bands and df_snap['ci_lower'].notna().any():
                # Convert color to rgba for transparency
                color_str = colors[i]
                if color_str.startswith('#'):
                    # Hex color - convert to rgba
                    rgb = tuple(int(color_str[j:j+2], 16) for j in (1, 3, 5))
                    fillcolor = f'rgba{rgb + (0.2,)}'
                elif color_str.startswith('rgb'):
                    # RGB color string
                    fillcolor = color_str.replace('rgb', 'rgba').replace(')', ', 0.2)')
                else:
                    fillcolor = 'rgba(128, 128, 128, 0.2)'
                
                fig.add_trace(
                    go.Scatter(
                        x=df_snap['bin'].astype(str),
                        y=df_snap['ci_upper'],
                        mode='lines',
                        name=f'CI Upper {snapshot}',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_snap['bin'].astype(str),
                        y=df_snap['ci_lower'],
                        mode='lines',
                        name=f'CI Lower {snapshot}',
                        fill='tonexty',
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=1, col=1,
                )
        
        # Plot 2: Population % by snapshot
        for i, snapshot in enumerate(snapshots):
            df_snap = df_agg[df_agg['snapshot'] == snapshot]
            df_snap = df_snap.sort_values('bin')
            
            fig.add_trace(
                go.Bar(
                    x=df_snap['bin'].astype(str),
                    y=df_snap['pop_pct'],
                    name=f'Pop % {snapshot}',
                    marker_color=colors[i],
                    opacity=0.7,
                ),
                row=2, col=1,
            )
        
        # Update layout
        plot_title = title or f"Temporal Bin Drift: {feature_col}"
        fig.update_layout(
            title_text=plot_title,
            height=800,
            template="plotly_white",
            hovermode='x unified',
        )
        
        fig.update_xaxes(title_text="Bin", row=2, col=1)
        fig.update_yaxes(title_text="Event Rate", row=1, col=1)
        fig.update_yaxes(title_text="Population %", row=2, col=1)
        
        return fig
    
    def plot_bin_delta_vs_baseline(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        snapshot_col: str,
        baseline_snapshot: str,
        comparison_snapshots: Optional[List] = None,
        segment_col: Optional[str] = None,
        max_bins_display: int = 20,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot delta (change) in event rate and population % vs baseline snapshot.
        
        Args:
            df: DataFrame with feature, target, and snapshot columns
            feature_col: Name of binned feature column
            target_col: Name of target column
            snapshot_col: Name of snapshot/time column
            baseline_snapshot: Reference snapshot
            comparison_snapshots: Snapshots to compare (None = all except baseline)
            segment_col: Optional segmentation column
            max_bins_display: Maximum number of bins to display
            title: Plot title
            
        Returns:
            Plotly figure showing delta vs baseline
            
        Example:
            >>> fig = visualizer.plot_bin_delta_vs_baseline(
            ...     df, "age_bin", "default", "month_end",
            ...     baseline_snapshot="2024-01"
            ... )
        """
        # Get available snapshots
        available_snapshots = sorted(df[snapshot_col].dropna().unique())
        
        # Handle missing baseline - default to first snapshot
        if baseline_snapshot not in available_snapshots:
            baseline_snapshot = available_snapshots[0] if available_snapshots else None
        
        if baseline_snapshot is None or len(available_snapshots) == 0:
            raise ValueError("No snapshots available in data")
        
        # Get baseline data
        df_baseline = df[df[snapshot_col] == baseline_snapshot].copy()
        
        # Get comparison snapshots
        if comparison_snapshots is None:
            comparison_snapshots = [s for s in available_snapshots if s != baseline_snapshot]
        
        # Get unique bins
        bins = sorted(df[feature_col].dropna().unique())
        if len(bins) > max_bins_display:
            bins = bins[:max_bins_display]
        
        # Calculate baseline metrics
        baseline_stats = df_baseline.groupby(feature_col)[target_col].agg(['count', 'sum'])
        baseline_stats['event_rate'] = baseline_stats['sum'] / baseline_stats['count']
        baseline_stats['pop_pct'] = baseline_stats['count'] / len(df_baseline)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Δ Event Rate vs Baseline", "Δ Population % vs Baseline"),
            horizontal_spacing=0.15,
        )
        
        colors = plotly.colors.qualitative.Set3[:len(comparison_snapshots)]
        
        for i, snapshot in enumerate(comparison_snapshots):
            df_snap = df[df[snapshot_col] == snapshot].copy()
            snap_stats = df_snap.groupby(feature_col)[target_col].agg(['count', 'sum'])
            snap_stats['event_rate'] = snap_stats['sum'] / snap_stats['count']
            snap_stats['pop_pct'] = snap_stats['count'] / len(df_snap)
            
            # Calculate deltas
            delta_event_rate = []
            delta_pop_pct = []
            bin_labels = []
            
            for bin_val in bins:
                if bin_val in baseline_stats.index and bin_val in snap_stats.index:
                    delta_er = snap_stats.loc[bin_val, 'event_rate'] - baseline_stats.loc[bin_val, 'event_rate']
                    delta_pop = snap_stats.loc[bin_val, 'pop_pct'] - baseline_stats.loc[bin_val, 'pop_pct']
                    
                    delta_event_rate.append(delta_er)
                    delta_pop_pct.append(delta_pop)
                    bin_labels.append(bin_val)
            
            # Plot deltas
            fig.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=delta_event_rate,
                    name=f'Δ Event Rate {snapshot}',
                    marker_color=colors[i],
                    opacity=0.7,
                ),
                row=1, col=1,
            )
            
            fig.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=delta_pop_pct,
                    name=f'Δ Pop % {snapshot}',
                    marker_color=colors[i],
                    opacity=0.7,
                ),
                row=1, col=2,
            )
        
        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        plot_title = title or f"Bin Delta vs Baseline ({baseline_snapshot}): {feature_col}"
        fig.update_layout(
            title_text=plot_title,
            height=500,
            template="plotly_white",
            barmode='group',
        )
        
        fig.update_xaxes(title_text="Bin", row=1, col=1)
        fig.update_xaxes(title_text="Bin", row=1, col=2)
        fig.update_yaxes(title_text="Δ Event Rate", row=1, col=1)
        fig.update_yaxes(title_text="Δ Population %", row=1, col=2)
        
        return fig
    
    def plot_psi_by_feature(
        self,
        df: pd.DataFrame,
        feature_col: str,
        snapshot_col: str,
        baseline_snapshot: str,
        comparison_snapshots: Optional[List] = None,
        n_bins: int = 10,
        segment_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot Population Stability Index (PSI) by feature across snapshots.
        
        Args:
            df: DataFrame with feature and snapshot columns
            feature_col: Name of feature column
            snapshot_col: Name of snapshot/time column
            baseline_snapshot: Reference snapshot
            comparison_snapshots: Snapshots to compare (None = all except baseline)
            n_bins: Number of bins for PSI calculation
            segment_col: Optional segmentation column
            title: Plot title
            
        Returns:
            Plotly figure showing PSI over time
            
        Example:
            >>> fig = visualizer.plot_psi_by_feature(
            ...     df, "age", "month_end", "2024-01"
            ... )
        """
        from cr_score.features.enhanced_features import FeatureValidator
        
        # Get baseline distribution
        df_baseline = df[df[snapshot_col] == baseline_snapshot]
        baseline_dist = df_baseline[feature_col].dropna()
        
        if comparison_snapshots is None:
            comparison_snapshots = [s for s in df[snapshot_col].unique() if s != baseline_snapshot]
        
        validator = FeatureValidator()
        
        # Calculate PSI for each snapshot
        psi_values = []
        snapshots_plot = []
        
        for snapshot in comparison_snapshots:
            df_snap = df[df[snapshot_col] == snapshot]
            current_dist = df_snap[feature_col].dropna()
            
            if len(current_dist) > 0:
                psi = validator.compute_psi(baseline_dist, current_dist, bins=n_bins)
                psi_values.append(psi)
                snapshots_plot.append(str(snapshot))
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=snapshots_plot,
                y=psi_values,
                mode='lines+markers',
                name='PSI',
                line=dict(color='red', width=2),
                marker=dict(size=10),
            )
        )
        
        # Add PSI threshold lines
        fig.add_hline(y=0.10, line_dash="dash", line_color="green", annotation_text="Low (<0.10)")
        fig.add_hline(y=0.25, line_dash="dash", line_color="orange", annotation_text="Medium (0.10-0.25)")
        fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="High (>0.25)", line_width=2)
        
        plot_title = title or f"PSI by Snapshot: {feature_col} (Baseline: {baseline_snapshot})"
        fig.update_layout(
            title_text=plot_title,
            xaxis_title="Snapshot",
            yaxis_title="PSI",
            template="plotly_white",
            height=500,
        )
        
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
            # Add metadata as annotation in layout
            metadata_text = "<br>".join([f"<b>{k}:</b> {v}" for k, v in metadata.items()])
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=-0.1,
                xanchor="left", yanchor="top",
                text=metadata_text,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
            
            # Update layout to accommodate metadata
            fig.update_layout(
                margin=dict(b=100),
            )
        
        if format == "html":
            fig.write_html(path)
        elif format == "png":
            fig.write_image(path, width=1200, height=800, scale=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported figure to {path}", metadata=metadata)
