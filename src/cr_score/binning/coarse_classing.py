"""
Coarse classing for final scorecard bins.

Merges fine bins based on monotonicity constraints and minimum bin size.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class CoarseClasser:
    """
    Merge fine bins into coarse bins for scorecard.

    Enforces monotonicity and minimum bin size constraints.

    Example:
        >>> classer = CoarseClasser(monotonicity=True, min_bin_pct=0.05)
        >>> coarse_bins = classer.fit_transform(df_binned, "age_bin", "target")
    """

    def __init__(
        self,
        monotonicity: bool = True,
        min_bin_pct: float = 0.05,
        min_bin_count: int = 30,
    ) -> None:
        """
        Initialize coarse classer.

        Args:
            monotonicity: Enforce monotonic event rate trend
            min_bin_pct: Minimum proportion of samples per bin
            min_bin_count: Minimum absolute count per bin
        """
        self.monotonicity = monotonicity
        self.min_bin_pct = min_bin_pct
        self.min_bin_count = min_bin_count
        self.logger = get_audit_logger()

        self.bin_mapping_: Optional[Dict[str, str]] = None
        self.binning_table_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        bin_col: str,
        target_col: str,
    ) -> "CoarseClasser":
        """
        Fit coarse binning scheme.

        Args:
            df: DataFrame with fine bins
            bin_col: Column with fine bin assignments
            target_col: Target variable

        Returns:
            Self

        Example:
            >>> classer.fit(df, "age_bin", "default")
        """
        self.logger.info(
            f"Fitting coarse classer for {bin_col}",
            monotonicity=self.monotonicity,
            min_bin_pct=self.min_bin_pct,
        )

        # Calculate fine bin statistics
        binning_table = self._calculate_bin_stats(df, bin_col, target_col)

        # Merge bins
        coarse_binning_table = self._merge_bins(binning_table, len(df))

        # Create mapping
        self.bin_mapping_ = dict(zip(
            coarse_binning_table["fine_bin"],
            coarse_binning_table["coarse_bin"]
        ))

        self.binning_table_ = coarse_binning_table

        self.is_fitted_ = True

        self.logger.info(
            f"Coarse classing fitted",
            fine_bins=len(binning_table),
            coarse_bins=coarse_binning_table["coarse_bin"].nunique(),
        )

        return self

    def _calculate_bin_stats(
        self,
        df: pd.DataFrame,
        bin_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Calculate statistics for each fine bin."""
        stats = df.groupby(bin_col, dropna=False).agg(
            count=(target_col, "count"),
            events=(target_col, "sum"),
        ).reset_index()

        stats["non_events"] = stats["count"] - stats["events"]
        stats["event_rate"] = stats["events"] / stats["count"]
        stats["pct"] = stats["count"] / stats["count"].sum()

        # Sort by bin order (handle categorical ordering)
        stats = stats.sort_values("event_rate")

        return stats

    def _merge_bins(
        self,
        binning_table: pd.DataFrame,
        total_count: int,
    ) -> pd.DataFrame:
        """Merge bins based on constraints."""
        # Start with each fine bin as separate coarse bin
        binning_table = binning_table.copy()
        binning_table["coarse_bin"] = range(len(binning_table))
        binning_table["fine_bin"] = binning_table.iloc[:, 0]

        # Merge small bins
        binning_table = self._merge_small_bins(binning_table, total_count)

        # Enforce monotonicity if required
        if self.monotonicity:
            binning_table = self._enforce_monotonicity(binning_table)

        # Renumber coarse bins
        coarse_bin_map = {old: new for new, old in enumerate(binning_table["coarse_bin"].unique())}
        binning_table["coarse_bin"] = binning_table["coarse_bin"].map(coarse_bin_map)

        # Recalculate statistics for coarse bins
        binning_table = self._recalculate_stats(binning_table)

        return binning_table

    def _merge_small_bins(
        self,
        binning_table: pd.DataFrame,
        total_count: int,
    ) -> pd.DataFrame:
        """Merge bins smaller than threshold."""
        min_count = max(self.min_bin_count, int(total_count * self.min_bin_pct))

        merged = []
        current_bin = []

        for idx, row in binning_table.iterrows():
            current_bin.append(row)

            # Check if current accumulated bin is large enough
            total_in_bin = sum(r["count"] for r in current_bin)

            if total_in_bin >= min_count:
                # Finalize this coarse bin
                coarse_idx = len(merged)
                for r in current_bin:
                    r["coarse_bin"] = coarse_idx
                    merged.append(r)
                current_bin = []

        # Handle remaining small bin
        if current_bin:
            if merged:
                # Merge with last coarse bin
                last_coarse_idx = merged[-1]["coarse_bin"]
                for r in current_bin:
                    r["coarse_bin"] = last_coarse_idx
                    merged.append(r)
            else:
                # All bins small, keep as one bin
                for r in current_bin:
                    r["coarse_bin"] = 0
                    merged.append(r)

        return pd.DataFrame(merged)

    def _enforce_monotonicity(
        self,
        binning_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enforce monotonic event rate trend."""
        # Group by coarse bin and check monotonicity
        coarse_stats = binning_table.groupby("coarse_bin").agg(
            count=("count", "sum"),
            events=("events", "sum"),
        )

        coarse_stats["event_rate"] = coarse_stats["events"] / coarse_stats["count"]

        # Check if monotonic
        event_rates = coarse_stats["event_rate"].values
        is_monotonic = all(event_rates[i] <= event_rates[i+1] for i in range(len(event_rates)-1))

        if is_monotonic:
            return binning_table

        # Merge non-monotonic adjacent bins
        # Simple approach: merge bins with decreasing event rate
        merged = []
        current_group = [0]

        for i in range(1, len(event_rates)):
            if event_rates[i] < event_rates[i-1]:
                # Non-monotonic, merge with previous
                current_group.append(i)
            else:
                # Monotonic, finalize previous group
                new_coarse_idx = len(merged) // 2 if merged else 0
                for idx in current_group:
                    merged.extend([new_coarse_idx] * len(binning_table[binning_table["coarse_bin"] == idx]))
                current_group = [i]

        # Finalize last group
        new_coarse_idx = len(set(merged))
        for idx in current_group:
            merged.extend([new_coarse_idx] * len(binning_table[binning_table["coarse_bin"] == idx]))

        binning_table["coarse_bin"] = merged

        return binning_table

    def _recalculate_stats(
        self,
        binning_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Recalculate statistics after merging."""
        # Group by coarse bin
        coarse_stats = binning_table.groupby("coarse_bin").agg(
            fine_bins=("fine_bin", lambda x: list(x)),
            count=("count", "sum"),
            events=("events", "sum"),
            non_events=("non_events", "sum"),
        ).reset_index()

        coarse_stats["event_rate"] = coarse_stats["events"] / coarse_stats["count"]
        coarse_stats["non_event_rate"] = coarse_stats["non_events"] / coarse_stats["count"]
        coarse_stats["pct"] = coarse_stats["count"] / coarse_stats["count"].sum()

        # Expand back to fine bin level for mapping
        expanded = []
        for _, row in coarse_stats.iterrows():
            for fine_bin in row["fine_bins"]:
                expanded.append({
                    "fine_bin": fine_bin,
                    "coarse_bin": f"Bin_{row['coarse_bin']+1}",
                    "count": row["count"],
                    "events": row["events"],
                    "event_rate": row["event_rate"],
                })

        return pd.DataFrame(expanded)

    def transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """
        Transform fine bins to coarse bins.

        Args:
            series: Series with fine bin assignments

        Returns:
            Series with coarse bin assignments

        Example:
            >>> df["age_coarse_bin"] = classer.transform(df["age_bin"])
        """
        if not self.is_fitted_:
            raise ValueError("Classer not fitted")

        return series.map(self.bin_mapping_)

    def fit_transform(
        self,
        df: pd.DataFrame,
        bin_col: str,
        target_col: str,
    ) -> pd.Series:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame with fine bins
            bin_col: Column with fine bin assignments
            target_col: Target variable

        Returns:
            Series with coarse bin assignments

        Example:
            >>> df["age_coarse"] = classer.fit_transform(df, "age_bin", "default")
        """
        return self.fit(df, bin_col, target_col).transform(df[bin_col])

    def get_binning_table(self) -> pd.DataFrame:
        """
        Get binning table with statistics.

        Returns:
            DataFrame with bin statistics

        Example:
            >>> binning_table = classer.get_binning_table()
            >>> print(binning_table[["coarse_bin", "count", "event_rate"]])
        """
        if not self.is_fitted_:
            raise ValueError("Classer not fitted")

        return self.binning_table_
