"""
Monotonic merge algorithm for binning.

Specialized algorithm to ensure monotonic event rate trend.
"""

from typing import List, Tuple

import pandas as pd

from cr_score.core.logging import get_audit_logger


class MonotonicMerger:
    """
    Enforce monotonic event rate trend through intelligent bin merging.

    Uses iterative merging to achieve monotonicity while preserving
    as much granularity as possible.

    Example:
        >>> merger = MonotonicMerger(direction="increasing")
        >>> binning_table_monotonic = merger.merge(binning_table)
    """

    def __init__(self, direction: str = "auto") -> None:
        """
        Initialize monotonic merger.

        Args:
            direction: Direction of monotonicity (increasing, decreasing, auto)
        """
        self.direction = direction
        self.logger = get_audit_logger()

    def merge(
        self,
        binning_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge bins to achieve monotonicity.

        Args:
            binning_table: Binning table with columns: bin, count, events, event_rate

        Returns:
            Merged binning table with monotonic event rates

        Example:
            >>> monotonic_table = merger.merge(binning_table)
        """
        self.logger.info("Starting monotonic merge")

        # Determine direction if auto
        if self.direction == "auto":
            direction = self._determine_direction(binning_table)
        else:
            direction = self.direction

        # Iteratively merge until monotonic
        result = binning_table.copy()
        iteration = 0

        while not self._is_monotonic(result, direction):
            result = self._merge_one_violation(result, direction)
            iteration += 1

            if iteration > 100:
                self.logger.warning("Max iterations reached in monotonic merge")
                break

        self.logger.info(
            f"Monotonic merge completed",
            direction=direction,
            iterations=iteration,
            final_bins=len(result),
        )

        return result

    def _determine_direction(self, binning_table: pd.DataFrame) -> str:
        """Determine monotonic direction from data."""
        event_rates = binning_table["event_rate"].values

        # Count violations in each direction
        increasing_violations = sum(
            event_rates[i] > event_rates[i+1]
            for i in range(len(event_rates)-1)
        )

        decreasing_violations = sum(
            event_rates[i] < event_rates[i+1]
            for i in range(len(event_rates)-1)
        )

        return "increasing" if increasing_violations < decreasing_violations else "decreasing"

    def _is_monotonic(self, binning_table: pd.DataFrame, direction: str) -> bool:
        """Check if event rates are monotonic."""
        event_rates = binning_table["event_rate"].values

        if direction == "increasing":
            return all(event_rates[i] <= event_rates[i+1] for i in range(len(event_rates)-1))
        else:
            return all(event_rates[i] >= event_rates[i+1] for i in range(len(event_rates)-1))

    def _merge_one_violation(
        self,
        binning_table: pd.DataFrame,
        direction: str,
    ) -> pd.DataFrame:
        """Merge one pair of bins that violates monotonicity."""
        event_rates = binning_table["event_rate"].values

        # Find first violation
        violation_idx = None

        if direction == "increasing":
            for i in range(len(event_rates)-1):
                if event_rates[i] > event_rates[i+1]:
                    violation_idx = i
                    break
        else:
            for i in range(len(event_rates)-1):
                if event_rates[i] < event_rates[i+1]:
                    violation_idx = i
                    break

        if violation_idx is None:
            return binning_table

        # Merge bins at violation_idx and violation_idx+1
        merged = []

        for i in range(len(binning_table)):
            if i < violation_idx:
                merged.append(binning_table.iloc[i])
            elif i == violation_idx:
                # Merge this bin with next
                row1 = binning_table.iloc[i]
                row2 = binning_table.iloc[i+1]

                merged_row = {
                    "bin": f"{row1['bin']}_merged_{row2['bin']}",
                    "count": row1["count"] + row2["count"],
                    "events": row1["events"] + row2["events"],
                }
                merged_row["event_rate"] = merged_row["events"] / merged_row["count"]

                merged.append(merged_row)
            elif i == violation_idx + 1:
                # Skip (already merged with previous)
                continue
            else:
                merged.append(binning_table.iloc[i])

        return pd.DataFrame(merged)
