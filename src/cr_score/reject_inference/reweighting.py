"""
Reweighting reject inference method.

Adjusts sample weights to account for selection bias from rejections.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cr_score.core.logging import get_audit_logger


class ReweightingInference:
    """
    Reweighting method for reject inference.

    Estimates probability of acceptance and reweights samples
    to correct for selection bias.

    Example:
        >>> inferencer = ReweightingInference()
        >>> df_reweighted = inferencer.fit_transform(df_all, "was_accepted")
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize reweighting inferencer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = get_audit_logger()

        self.propensity_model_: Optional[LogisticRegression] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        acceptance_col: str,
        feature_cols: list,
    ) -> "ReweightingInference":
        """
        Fit propensity model to estimate probability of acceptance.

        Args:
            df: DataFrame with both accepts and rejects
            acceptance_col: Binary column (1=accepted, 0=rejected)
            feature_cols: Features to use for propensity modeling

        Returns:
            Self

        Example:
            >>> inferencer.fit(df, "was_accepted", ["age", "income", "credit_score"])
        """
        self.logger.info(
            "Fitting propensity model for reweighting",
            n_samples=len(df),
            acceptance_rate=df[acceptance_col].mean(),
        )

        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[acceptance_col]

        # Fit logistic regression for propensity scores
        self.propensity_model_ = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
        )

        self.propensity_model_.fit(X, y)

        self.is_fitted_ = True

        self.logger.info("Propensity model fitted")

        return self

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> pd.DataFrame:
        """
        Calculate propensity weights.

        Args:
            df: DataFrame
            feature_cols: Features used in propensity model
            min_weight: Minimum weight cap
            max_weight: Maximum weight cap

        Returns:
            DataFrame with added propensity_weight column

        Example:
            >>> df_weighted = inferencer.transform(df, ["age", "income"])
        """
        if not self.is_fitted_:
            raise ValueError("Inferencer not fitted")

        # Predict propensity scores
        X = df[feature_cols].fillna(0)
        propensity_scores = self.propensity_model_.predict_proba(X)[:, 1]

        # Calculate inverse propensity weights
        # Weight = 1 / P(accept)
        weights = 1.0 / (propensity_scores + 0.001)  # Add epsilon for stability

        # Cap weights
        weights = np.clip(weights, min_weight, max_weight)

        # Add to dataframe
        df_out = df.copy()
        df_out["propensity_weight"] = weights
        df_out["propensity_score"] = propensity_scores

        self.logger.info(
            "Propensity weights calculated",
            mean_weight=float(weights.mean()),
            min_weight_applied=float(weights.min()),
            max_weight_applied=float(weights.max()),
        )

        return df_out

    def fit_transform(
        self,
        df: pd.DataFrame,
        acceptance_col: str,
        feature_cols: list,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame with accepts and rejects
            acceptance_col: Acceptance indicator
            feature_cols: Features for propensity model
            min_weight: Minimum weight
            max_weight: Maximum weight

        Returns:
            DataFrame with propensity weights

        Example:
            >>> df_weighted = inferencer.fit_transform(
            ...     df_all,
            ...     "was_accepted",
            ...     ["age", "income", "credit_score"]
            ... )
        """
        return self.fit(df, acceptance_col, feature_cols).transform(
            df, feature_cols, min_weight, max_weight
        )

    def get_propensity_stats(self, df: pd.DataFrame) -> dict:
        """
        Get propensity score statistics.

        Args:
            df: DataFrame with propensity_score column

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = inferencer.get_propensity_stats(df_weighted)
            >>> print(stats)
        """
        if "propensity_score" not in df.columns:
            raise ValueError("DataFrame missing propensity_score column")

        scores = df["propensity_score"]

        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "q25": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "q75": float(scores.quantile(0.75)),
            "max": float(scores.max()),
        }
