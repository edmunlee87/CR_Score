"""
Binning visualization for scorecard analysis.

Creates interactive plots for bin analysis using Plotly.
"""

from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
