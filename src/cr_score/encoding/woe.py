"""
Weight of Evidence (WoE) encoding for scorecard variables.

Transforms binned categorical variables into continuous WoE values
for use in logistic regression.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class WoEEncoder:
    """
    Calculate and apply Weight of Evidence encoding.

    WoE Formula:
        WoE = ln(% non-events / % events)

    Information Value (IV):
        IV = Σ [(% non-events - % events) * WoE]

    Example:
        >>> encoder = WoEEncoder()
        >>> encoder.fit(df["age_bin"], df["target"])
        >>> df["age_woe"] = encoder.transform(df["age_bin"])
        >>> print(f"IV: {encoder.iv_:.3f}")
    """

    def __init__(self, epsilon: float = 0.0001) -> None:
        """
        Initialize WoE encoder.

        Args:
            epsilon: Small constant to avoid log(0) and division by zero
        """
        self.epsilon = epsilon
        self.logger = get_audit_logger()

        self.woe_map_: Optional[Dict[Any, float]] = None
        self.iv_: Optional[float] = None
        self.woe_table_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> "WoEEncoder":
        """
        Calculate WoE values for each bin.

        Args:
            X: Binned categorical variable
            y: Binary target variable (0/1)

        Returns:
            Self

        Example:
            >>> encoder.fit(df["income_bin"], df["default"])
        """
        feature_name = X.name if hasattr(X, "name") else "feature"

        self.logger.info(f"Calculating WoE for {feature_name}")

        # Calculate bin statistics
        stats = pd.DataFrame({
            "bin": X,
            "target": y,
        }).groupby("bin", dropna=False).agg(
            total_count=("target", "count"),
            event_count=("target", "sum"),
        ).reset_index()

        stats["non_event_count"] = stats["total_count"] - stats["event_count"]

        # Calculate overall distributions
        total_events = stats["event_count"].sum()
        total_non_events = stats["non_event_count"].sum()

        # Calculate percentages with epsilon to avoid division by zero
        stats["event_pct"] = (stats["event_count"] + self.epsilon) / (total_events + self.epsilon)
        stats["non_event_pct"] = (stats["non_event_count"] + self.epsilon) / (total_non_events + self.epsilon)

        # Calculate WoE
        stats["woe"] = np.log(stats["non_event_pct"] / stats["event_pct"])

        # Calculate event rate
        stats["event_rate"] = stats["event_count"] / stats["total_count"]

        # Calculate IV contribution
        stats["iv_contribution"] = (stats["non_event_pct"] - stats["event_pct"]) * stats["woe"]

        # Total IV
        self.iv_ = stats["iv_contribution"].sum()

        # Create mapping
        self.woe_map_ = dict(zip(stats["bin"], stats["woe"]))

        # Store table
        self.woe_table_ = stats[[
            "bin", "total_count", "event_count", "non_event_count",
            "event_rate", "woe", "iv_contribution"
        ]]

        self.is_fitted_ = True

        self.logger.info(
            f"WoE calculation completed",
            n_bins=len(stats),
            iv=self.iv_,
            predictive_power=self._interpret_iv(self.iv_),
        )

        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform bins to WoE values.

        Args:
            X: Binned variable

        Returns:
            Series with WoE values

        Example:
            >>> df["income_woe"] = encoder.transform(df["income_bin"])
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted. Call fit() first.")

        return X.map(self.woe_map_).fillna(0.0)

    def fit_transform(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> pd.Series:
        """
        Fit and transform in one step.

        Args:
            X: Binned variable
            y: Binary target

        Returns:
            Series with WoE values

        Example:
            >>> df["age_woe"] = encoder.fit_transform(df["age_bin"], df["default"])
        """
        return self.fit(X, y).transform(X)

    def get_woe_table(self) -> pd.DataFrame:
        """
        Get WoE table with statistics.

        Returns:
            DataFrame with WoE values and IV contributions per bin

        Example:
            >>> woe_table = encoder.get_woe_table()
            >>> print(woe_table.to_string())
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        return self.woe_table_.copy()

    def get_iv(self) -> float:
        """
        Get Information Value.

        Returns:
            IV value

        Example:
            >>> iv = encoder.get_iv()
            >>> print(f"IV: {iv:.3f} - {encoder.interpret_iv()}")
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        return self.iv_

    def interpret_iv(self) -> str:
        """
        Interpret IV value.

        Returns:
            Human-readable interpretation

        Example:
            >>> print(encoder.interpret_iv())
            'Strong predictor'
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        return self._interpret_iv(self.iv_)

    def _interpret_iv(self, iv: float) -> str:
        """Interpret IV value."""
        if iv < 0.02:
            return "Not useful for prediction"
        elif iv < 0.1:
            return "Weak predictor"
        elif iv < 0.3:
            return "Medium predictor"
        elif iv < 0.5:
            return "Strong predictor"
        else:
            return "Suspicious (overfitting risk)"

    def get_mapping_dict(self) -> Dict[Any, float]:
        """
        Get bin-to-WoE mapping dictionary.

        Returns:
            Dictionary mapping bin labels to WoE values

        Example:
            >>> mapping = encoder.get_mapping_dict()
            >>> print(mapping)
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        return self.woe_map_.copy()

    def export_to_json(self) -> Dict[str, Any]:
        """
        Export WoE mapping to JSON-serializable format.

        Returns:
            Dictionary with WoE mapping and metadata

        Example:
            >>> json_data = encoder.export_to_json()
            >>> import json
            >>> with open("woe_mapping.json", "w") as f:
            ...     json.dump(json_data, f)
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        return {
            "woe_mapping": {str(k): float(v) for k, v in self.woe_map_.items()},
            "iv": float(self.iv_),
            "predictive_power": self.interpret_iv(),
            "n_bins": len(self.woe_map_),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "WoEEncoder":
        """
        Load WoE encoder from JSON data.

        Args:
            json_data: Dictionary from export_to_json()

        Returns:
            Fitted WoEEncoder instance

        Example:
            >>> import json
            >>> with open("woe_mapping.json") as f:
            ...     json_data = json.load(f)
            >>> encoder = WoEEncoder.from_json(json_data)
        """
        encoder = cls(epsilon=json_data.get("epsilon", 0.0001))
        encoder.woe_map_ = json_data["woe_mapping"]
        encoder.iv_ = json_data["iv"]
        encoder.is_fitted_ = True

        return encoder


class MultiWoEEncoder:
    """
    Encode multiple features with WoE.

    Convenience class for encoding all scorecard features.

    Example:
        >>> encoder = MultiWoEEncoder()
        >>> woe_df = encoder.fit_transform(df_binned, target_col="default")
        >>> iv_summary = encoder.get_iv_summary()
    """

    def __init__(self, epsilon: float = 0.0001) -> None:
        """
        Initialize multi-feature WoE encoder.

        Args:
            epsilon: Small constant for WoE calculation
        """
        self.epsilon = epsilon
        self.logger = get_audit_logger()

        self.encoders_: Dict[str, WoEEncoder] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
    ) -> "MultiWoEEncoder":
        """
        Fit WoE encoders for all features.

        Args:
            df: DataFrame with binned features
            target_col: Target variable column
            feature_cols: List of feature columns (None = all except target)

        Returns:
            Self

        Example:
            >>> encoder.fit(df_binned, target_col="default")
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        self.logger.info(f"Fitting WoE encoders for {len(feature_cols)} features")

        for feature in feature_cols:
            encoder = WoEEncoder(epsilon=self.epsilon)
            encoder.fit(df[feature], df[target_col])
            self.encoders_[feature] = encoder

        self.is_fitted_ = True

        self.logger.info("Multi-feature WoE encoding completed")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to WoE values.

        Args:
            df: DataFrame with binned features

        Returns:
            DataFrame with WoE-encoded features

        Example:
            >>> df_woe = encoder.transform(df_binned)
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        df_woe = pd.DataFrame(index=df.index)

        for feature, encoder in self.encoders_.items():
            if feature in df.columns:
                df_woe[f"{feature}_woe"] = encoder.transform(df[feature])

        return df_woe

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame with binned features
            target_col: Target variable
            feature_cols: Features to encode

        Returns:
            DataFrame with WoE-encoded features

        Example:
            >>> df_woe = encoder.fit_transform(df_binned, "default")
        """
        return self.fit(df, target_col, feature_cols).transform(df)

    def get_iv_summary(self) -> pd.DataFrame:
        """
        Get IV summary for all features.

        Returns:
            DataFrame with IV values and interpretations

        Example:
            >>> iv_summary = encoder.get_iv_summary()
            >>> print(iv_summary.sort_values("iv", ascending=False))
        """
        if not self.is_fitted_:
            raise ValueError("Encoder not fitted")

        rows = []
        for feature, encoder in self.encoders_.items():
            rows.append({
                "feature": feature,
                "iv": encoder.get_iv(),
                "predictive_power": encoder.interpret_iv(),
                "n_bins": len(encoder.woe_map_),
            })

        return pd.DataFrame(rows).sort_values("iv", ascending=False)
