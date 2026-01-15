"""
Parceling reject inference method.

Assigns outcomes to rejects based on score distribution matching.
"""

from typing import Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class ParcelingInference:
    """
    Parceling method for reject inference.

    Assigns predicted outcomes to rejected applications by matching
    score distributions between accepts and rejects.

    Example:
        >>> inferencer = ParcelingInference(bad_rate=0.10)
        >>> df_with_rejects = inferencer.infer(df_accepts, df_rejects, score_col="score")
    """

    def __init__(
        self,
        bad_rate: float = 0.10,
        random_state: int = 42,
    ) -> None:
        """
        Initialize parceling inferencer.

        Args:
            bad_rate: Expected bad rate for rejects
            random_state: Random seed for reproducibility
        """
        self.bad_rate = bad_rate
        self.random_state = random_state
        self.logger = get_audit_logger()

    def infer(
        self,
        df_accepts: pd.DataFrame,
        df_rejects: pd.DataFrame,
        score_col: str,
        target_col: str = "target",
    ) -> pd.DataFrame:
        """
        Infer outcomes for rejected applications.

        Args:
            df_accepts: Accepted applications with known outcomes
            df_rejects: Rejected applications
            score_col: Score column name
            target_col: Target column name

        Returns:
            Combined DataFrame with inferred reject outcomes

        Example:
            >>> df_combined = inferencer.infer(
            ...     df_accepts,
            ...     df_rejects,
            ...     score_col="application_score"
            ... )
        """
        self.logger.info(
            "Starting parceling reject inference",
            n_accepts=len(df_accepts),
            n_rejects=len(df_rejects),
            bad_rate=self.bad_rate,
        )

        # Sort rejects by score (worst to best)
        df_rejects_sorted = df_rejects.sort_values(score_col).copy()

        # Assign bad outcomes to lowest scoring rejects
        n_bad = int(len(df_rejects_sorted) * self.bad_rate)

        df_rejects_sorted[target_col] = 0
        df_rejects_sorted.iloc[:n_bad, df_rejects_sorted.columns.get_loc(target_col)] = 1

        # Add source indicator
        df_accepts_marked = df_accepts.copy()
        df_accepts_marked["_source"] = "accept"

        df_rejects_marked = df_rejects_sorted.copy()
        df_rejects_marked["_source"] = "reject_inferred"

        # Combine
        df_combined = pd.concat([df_accepts_marked, df_rejects_marked], ignore_index=True)

        self.logger.info(
            "Parceling inference completed",
            total_samples=len(df_combined),
            inferred_bads=n_bad,
            inferred_bad_rate=n_bad / len(df_rejects_sorted) if len(df_rejects_sorted) > 0 else 0,
        )

        return df_combined

    def validate_assumptions(
        self,
        df_accepts: pd.DataFrame,
        df_rejects: pd.DataFrame,
        score_col: str,
    ) -> dict:
        """
        Validate reject inference assumptions.

        Args:
            df_accepts: Accepted applications
            df_rejects: Rejected applications
            score_col: Score column

        Returns:
            Dictionary with validation metrics

        Example:
            >>> validation = inferencer.validate_assumptions(df_accepts, df_rejects, "score")
            >>> if validation["score_overlap"] < 0.3:
            ...     print("WARNING: Low score overlap between accepts and rejects")
        """
        accepts_scores = df_accepts[score_col].dropna()
        rejects_scores = df_rejects[score_col].dropna()

        # Score distribution overlap
        accepts_range = (accepts_scores.min(), accepts_scores.max())
        rejects_range = (rejects_scores.min(), rejects_scores.max())

        overlap_min = max(accepts_range[0], rejects_range[0])
        overlap_max = min(accepts_range[1], rejects_range[1])

        if overlap_max > overlap_min:
            overlap_range = overlap_max - overlap_min
            total_range = max(accepts_range[1], rejects_range[1]) - min(accepts_range[0], rejects_range[0])
            overlap_ratio = overlap_range / total_range if total_range > 0 else 0
        else:
            overlap_ratio = 0.0

        return {
            "score_overlap": float(overlap_ratio),
            "accepts_mean_score": float(accepts_scores.mean()),
            "rejects_mean_score": float(rejects_scores.mean()),
            "score_separation": float(abs(accepts_scores.mean() - rejects_scores.mean())),
            "accepts_range": accepts_range,
            "rejects_range": rejects_range,
        }
