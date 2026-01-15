"""
Fine classing for initial binning.

Creates granular bins using quantile, equal-width, or decision tree methods.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from cr_score.core.config.schema import BinningMethod
from cr_score.core.logging import get_audit_logger


class FineClasser:
    """
    Create fine-grained bins for numeric and categorical variables.

    Fine bins are later merged into coarse bins for final scorecard.

    Example:
        >>> classer = FineClasser(method="quantile", max_bins=20)
        >>> bins = classer.fit(df["age"], df["target"])
        >>> df["age_bin"] = classer.transform(df["age"])
    """

    def __init__(
        self,
        method: Union[str, BinningMethod] = BinningMethod.QUANTILE,
        max_bins: int = 20,
        min_bin_size: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """
        Initialize fine classer.

        Args:
            method: Binning method (quantile, equal_width, decision_tree, custom)
            max_bins: Maximum number of bins
            min_bin_size: Minimum proportion of samples per bin
            random_state: Random seed for reproducibility
        """
        if isinstance(method, str):
            method = BinningMethod(method)

        self.method = method
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.random_state = random_state
        self.logger = get_audit_logger()

        self.bin_edges_: Optional[np.ndarray] = None
        self.bin_labels_: Optional[List[str]] = None
        self.feature_name_: Optional[str] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: pd.Series,
        y: Optional[pd.Series] = None,
    ) -> "FineClasser":
        """
        Fit binning scheme to data.

        Args:
            X: Feature to bin
            y: Target variable (required for decision_tree method)

        Returns:
            Self

        Example:
            >>> classer.fit(df["income"], df["default"])
        """
        self.feature_name_ = X.name if hasattr(X, "name") else "feature"

        # Remove missing values
        if y is not None:
            valid_idx = X.notna() & y.notna()
            X_clean = X[valid_idx]
            y_clean = y[valid_idx] if y is not None else None
        else:
            X_clean = X.dropna()
            y_clean = None

        if len(X_clean) == 0:
            raise ValueError("No valid data after removing missing values")

        self.logger.info(
            f"Fitting fine classer for {self.feature_name_}",
            method=self.method.value,
            max_bins=self.max_bins,
            valid_samples=len(X_clean),
        )

        # Determine if numeric or categorical
        if pd.api.types.is_numeric_dtype(X_clean):
            self._fit_numeric(X_clean, y_clean)
        else:
            self._fit_categorical(X_clean, y_clean)

        self.is_fitted_ = True

        self.logger.info(
            f"Fine classing fitted",
            n_bins=len(self.bin_edges_) - 1 if self.bin_edges_ is not None else len(self.bin_labels_),
        )

        return self

    def _fit_numeric(
        self,
        X: pd.Series,
        y: Optional[pd.Series] = None,
    ) -> None:
        """Fit numeric variable binning."""
        if self.method == BinningMethod.QUANTILE:
            self.bin_edges_ = self._quantile_bins(X)
        elif self.method == BinningMethod.EQUAL_WIDTH:
            self.bin_edges_ = self._equal_width_bins(X)
        elif self.method == BinningMethod.DECISION_TREE:
            if y is None:
                raise ValueError("Target variable required for decision_tree method")
            self.bin_edges_ = self._decision_tree_bins(X, y)
        else:
            raise ValueError(f"Unknown binning method: {self.method}")

        # Create bin labels
        self.bin_labels_ = self._create_bin_labels(self.bin_edges_)

    def _fit_categorical(
        self,
        X: pd.Series,
        y: Optional[pd.Series] = None,
    ) -> None:
        """Fit categorical variable binning."""
        # For categorical, each unique value is a bin
        # Later merged in coarse classing
        value_counts = X.value_counts()

        # Keep top categories, group rare ones
        min_count = int(len(X) * self.min_bin_size)

        frequent_cats = value_counts[value_counts >= min_count].index.tolist()
        rare_cats = value_counts[value_counts < min_count].index.tolist()

        self.bin_labels_ = frequent_cats
        if rare_cats:
            self.bin_labels_.append("__OTHER__")

        self.logger.info(
            f"Categorical binning: {len(frequent_cats)} frequent + {len(rare_cats)} rare (grouped)"
        )

    def _quantile_bins(self, X: pd.Series) -> np.ndarray:
        """Create quantile-based bins."""
        _, edges = pd.qcut(X, q=self.max_bins, retbins=True, duplicates="drop")

        # Ensure edges cover full range
        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges

    def _equal_width_bins(self, X: pd.Series) -> np.ndarray:
        """Create equal-width bins."""
        _, edges = pd.cut(X, bins=self.max_bins, retbins=True)

        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges

    def _decision_tree_bins(
        self,
        X: pd.Series,
        y: pd.Series,
    ) -> np.ndarray:
        """Create bins using decision tree splits."""
        # Fit decision tree
        dt = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=int(len(X) * self.min_bin_size),
            random_state=self.random_state,
        )

        dt.fit(X.values.reshape(-1, 1), y.values)

        # Extract split thresholds
        tree = dt.tree_
        thresholds = []

        def extract_thresholds(node: int) -> None:
            if tree.feature[node] != -2:
                thresholds.append(tree.threshold[node])
                extract_thresholds(tree.children_left[node])
                extract_thresholds(tree.children_right[node])

        extract_thresholds(0)

        # Create bin edges
        edges = sorted(set(thresholds))
        edges = [-np.inf] + edges + [np.inf]

        return np.array(edges)

    def _create_bin_labels(self, edges: np.ndarray) -> List[str]:
        """Create human-readable bin labels."""
        labels = []

        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]

            if np.isinf(lower):
                label = f"< {upper:.2f}"
            elif np.isinf(upper):
                label = f">= {lower:.2f}"
            else:
                label = f"[{lower:.2f}, {upper:.2f})"

            labels.append(label)

        return labels

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Transform feature into bin assignments.

        Args:
            X: Feature to bin

        Returns:
            Series with bin labels

        Example:
            >>> df["age_bin"] = classer.transform(df["age"])
        """
        if not self.is_fitted_:
            raise ValueError("Classer not fitted. Call fit() first.")

        if pd.api.types.is_numeric_dtype(X):
            return pd.cut(X, bins=self.bin_edges_, labels=self.bin_labels_)
        else:
            # Categorical
            def map_category(val: Any) -> str:
                if pd.isna(val):
                    return "__MISSING__"
                elif val in self.bin_labels_:
                    return str(val)
                else:
                    return "__OTHER__"

            return X.apply(map_category)

    def fit_transform(
        self,
        X: pd.Series,
        y: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Fit and transform in one step.

        Args:
            X: Feature to bin
            y: Target variable

        Returns:
            Series with bin assignments

        Example:
            >>> df["income_bin"] = classer.fit_transform(df["income"], df["default"])
        """
        return self.fit(X, y).transform(X)

    def get_bin_info(self) -> Dict[str, Any]:
        """
        Get binning information.

        Returns:
            Dictionary with binning details

        Example:
            >>> info = classer.get_bin_info()
            >>> print(info["n_bins"])
        """
        if not self.is_fitted_:
            raise ValueError("Classer not fitted")

        return {
            "feature_name": self.feature_name_,
            "method": self.method.value,
            "n_bins": len(self.bin_labels_),
            "bin_labels": self.bin_labels_,
            "bin_edges": self.bin_edges_.tolist() if self.bin_edges_ is not None else None,
        }
