"""
OptBinning wrapper for automatic optimal binning.

Integrates the optbinning library for advanced optimal binning algorithms.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger

try:
    from optbinning import OptimalBinning
    OPTBINNING_AVAILABLE = True
except ImportError:
    OPTBINNING_AVAILABLE = False


class OptBinningWrapper:
    """
    Wrapper for OptimalBinning with simplified interface.

    Uses mathematical optimization to find best bins that maximize
    predictive power while respecting constraints.

    Example:
        >>> binner = OptBinningWrapper(max_n_bins=5, monotonic_trend="auto")
        >>> binner.fit(df["age"], df["target"])
        >>> df["age_bin"] = binner.transform(df["age"])
        >>> print(f"IV: {binner.get_iv():.3f}")
    """

    def __init__(
        self,
        max_n_bins: int = 5,
        min_bin_size: float = 0.05,
        monotonic_trend: str = "auto",
        min_event_rate_diff: float = 0.0,
    ) -> None:
        """
        Initialize OptBinning wrapper.

        Args:
            max_n_bins: Maximum number of bins
            min_bin_size: Minimum proportion of samples per bin
            monotonic_trend: Monotonicity direction (auto, ascending, descending, None)
            min_event_rate_diff: Minimum event rate difference between bins

        Raises:
            ImportError: If optbinning package not installed
        """
        if not OPTBINNING_AVAILABLE:
            raise ImportError(
                "optbinning package not installed. "
                "Install with: pip install optbinning"
            )

        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.logger = get_audit_logger()

        self.optb_: Optional[OptimalBinning] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> "OptBinningWrapper":
        """
        Fit optimal binning.

        Args:
            X: Feature to bin
            y: Binary target variable

        Returns:
            Self

        Example:
            >>> binner.fit(df["income"], df["default"])
        """
        feature_name = X.name if hasattr(X, "name") else "feature"

        self.logger.info(
            f"Fitting OptimalBinning for {feature_name}",
            max_n_bins=self.max_n_bins,
            monotonic_trend=self.monotonic_trend,
        )

        # Determine variable type
        if pd.api.types.is_numeric_dtype(X):
            dtype = "numerical"
        else:
            dtype = "categorical"

        # Create OptimalBinning instance
        self.optb_ = OptimalBinning(
            name=feature_name,
            dtype=dtype,
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            monotonic_trend=self.monotonic_trend if self.monotonic_trend != "None" else None,
            min_event_rate_diff=self.min_event_rate_diff,
        )

        # Fit
        self.optb_.fit(X.values, y.values)

        self.is_fitted_ = True

        self.logger.info(
            "OptimalBinning completed",
            n_bins=len(self.optb_.splits) + 1 if hasattr(self.optb_, 'splits') else self.max_n_bins,
            iv=self.get_iv(),
        )

        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform feature to bin labels.

        Args:
            X: Feature to bin

        Returns:
            Series with bin labels

        Example:
            >>> df["age_bin"] = binner.transform(df["age"])
        """
        if not self.is_fitted_:
            raise ValueError("Binner not fitted. Call fit() first.")

        return pd.Series(self.optb_.transform(X.values), index=X.index)

    def fit_transform(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> pd.Series:
        """
        Fit and transform in one step.

        Args:
            X: Feature to bin
            y: Binary target

        Returns:
            Series with bin labels

        Example:
            >>> df["income_bin"] = binner.fit_transform(df["income"], df["default"])
        """
        return self.fit(X, y).transform(X)

    def get_binning_table(self) -> pd.DataFrame:
        """
        Get binning table with statistics.

        Returns:
            DataFrame with bin statistics

        Example:
            >>> table = binner.get_binning_table()
            >>> print(table[["Bin", "Count", "Event rate", "WoE", "IV"]])
        """
        if not self.is_fitted_:
            raise ValueError("Binner not fitted")

        return self.optb_.binning_table.build()

    def get_iv(self) -> float:
        """
        Get Information Value.

        Returns:
            IV value

        Example:
            >>> iv = binner.get_iv()
            >>> print(f"IV: {iv:.3f}")
        """
        if not self.is_fitted_:
            raise ValueError("Binner not fitted")

        binning_table = self.get_binning_table()
        return float(binning_table["IV"].iloc[-1])

    def get_woe_transform(self, X: pd.Series) -> pd.Series:
        """
        Transform feature directly to WoE values.

        Args:
            X: Feature to transform

        Returns:
            Series with WoE values

        Example:
            >>> df["age_woe"] = binner.get_woe_transform(df["age"])
        """
        if not self.is_fitted_:
            raise ValueError("Binner not fitted")

        # OptBinning can transform directly to WoE
        woe_values = self.optb_.transform(X.values, metric="woe")
        return pd.Series(woe_values, index=X.index)


class AutoBinner:
    """
    Automatic binning for multiple features.

    Simplifies binning of all features in a dataset with optimal algorithms.

    Example:
        >>> auto_binner = AutoBinner(max_n_bins=5)
        >>> df_binned, df_woe = auto_binner.fit_transform(df, target_col="default")
        >>> iv_summary = auto_binner.get_iv_summary()
        >>> print(iv_summary.head())
    """

    def __init__(
        self,
        max_n_bins: int = 5,
        min_bin_size: float = 0.05,
        monotonic_trend: str = "auto",
        min_iv: float = 0.02,
    ) -> None:
        """
        Initialize AutoBinner.

        Args:
            max_n_bins: Maximum bins per feature
            min_bin_size: Minimum proportion per bin
            monotonic_trend: Monotonicity direction
            min_iv: Minimum IV to include feature
        """
        if not OPTBINNING_AVAILABLE:
            raise ImportError(
                "optbinning package not installed. "
                "Install with: pip install optbinning"
            )

        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.min_iv = min_iv
        self.logger = get_audit_logger()

        self.binners_: Dict[str, OptBinningWrapper] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
    ) -> "AutoBinner":
        """
        Fit binning for all features.

        Args:
            df: DataFrame with features and target
            target_col: Target column name
            feature_cols: Features to bin (None = all except target)

        Returns:
            Self

        Example:
            >>> auto_binner.fit(df, target_col="default")
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        self.logger.info(f"Auto-binning {len(feature_cols)} features")

        for feature in feature_cols:
            try:
                binner = OptBinningWrapper(
                    max_n_bins=self.max_n_bins,
                    min_bin_size=self.min_bin_size,
                    monotonic_trend=self.monotonic_trend,
                )

                binner.fit(df[feature], df[target_col])

                # Only keep if IV above threshold
                if binner.get_iv() >= self.min_iv:
                    self.binners_[feature] = binner
                    self.logger.info(
                        f"  {feature}: IV={binner.get_iv():.3f} - INCLUDED"
                    )
                else:
                    self.logger.info(
                        f"  {feature}: IV={binner.get_iv():.3f} - EXCLUDED (below min_iv)"
                    )

            except Exception as e:
                self.logger.warning(f"  {feature}: FAILED - {e}")

        self.is_fitted_ = True
        self.logger.info(f"Auto-binning completed: {len(self.binners_)} features selected")

        return self

    def transform_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to bin labels.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with binned features

        Example:
            >>> df_binned = auto_binner.transform_bins(df)
        """
        if not self.is_fitted_:
            raise ValueError("AutoBinner not fitted")

        df_binned = pd.DataFrame(index=df.index)

        for feature, binner in self.binners_.items():
            if feature in df.columns:
                df_binned[f"{feature}_bin"] = binner.transform(df[feature])

        return df_binned

    def transform_woe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features directly to WoE values.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with WoE-encoded features

        Example:
            >>> df_woe = auto_binner.transform_woe(df)
        """
        if not self.is_fitted_:
            raise ValueError("AutoBinner not fitted")

        df_woe = pd.DataFrame(index=df.index)

        for feature, binner in self.binners_.items():
            if feature in df.columns:
                df_woe[f"{feature}_woe"] = binner.get_woe_transform(df[feature])

        return df_woe

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame with features and target
            target_col: Target column
            feature_cols: Features to bin

        Returns:
            Tuple of (binned DataFrame, WoE DataFrame)

        Example:
            >>> df_binned, df_woe = auto_binner.fit_transform(df, "default")
        """
        self.fit(df, target_col, feature_cols)
        return self.transform_bins(df), self.transform_woe(df)

    def get_iv_summary(self) -> pd.DataFrame:
        """
        Get IV summary for all features.

        Returns:
            DataFrame with IV values sorted by strength

        Example:
            >>> iv_summary = auto_binner.get_iv_summary()
            >>> print(iv_summary)
        """
        if not self.is_fitted_:
            raise ValueError("AutoBinner not fitted")

        rows = []
        for feature, binner in self.binners_.items():
            rows.append({
                "feature": feature,
                "iv": binner.get_iv(),
                "n_bins": len(binner.optb_.splits) + 1 if hasattr(binner.optb_, 'splits') else self.max_n_bins,
            })

        return pd.DataFrame(rows).sort_values("iv", ascending=False)

    def get_selected_features(self) -> List[str]:
        """
        Get list of selected features.

        Returns:
            List of feature names

        Example:
            >>> features = auto_binner.get_selected_features()
            >>> print(f"Selected {len(features)} features")
        """
        return list(self.binners_.keys())
