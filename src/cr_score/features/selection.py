"""
Model-agnostic feature selection methods with MLflow tracking.

Supports: Forward, Backward, Stepwise, and Exhaustive selection.
Works with any scikit-learn compatible estimator.
"""

import warnings
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from cr_score.core.logging import get_audit_logger

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class BaseFeatureSelector(ABC):
    """
    Base class for feature selection methods.

    All selectors are model-agnostic and work with any sklearn-compatible estimator.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        scoring: str = "roc_auc",
        cv: int = 5,
        use_mlflow: bool = True,
        mlflow_experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize feature selector.

        Args:
            estimator: Any sklearn-compatible model (LogisticRegression, RandomForest, etc.)
            scoring: Scoring metric (roc_auc, accuracy, f1, etc.)
            cv: Number of cross-validation folds
            use_mlflow: Whether to use MLflow for experiment tracking
            mlflow_experiment_name: MLflow experiment name (auto-generated if None)
        """
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.use_mlflow = use_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        self.logger = get_audit_logger()

        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[Tuple, float]] = None
        self.best_score_: Optional[float] = None
        self.is_fitted_: bool = False

        # Setup MLflow
        if self.use_mlflow and MLFLOW_AVAILABLE:
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseFeatureSelector":
        """Fit feature selector."""
        pass

    def _evaluate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        log_to_mlflow: bool = True,
    ) -> float:
        """
        Evaluate feature subset using cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            features: List of features to evaluate
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Mean cross-validation score
        """
        if len(features) == 0:
            return 0.0

        X_subset = X[features]
        estimator = clone(self.estimator)

        # Cross-validate
        scores = cross_val_score(
            estimator,
            X_subset,
            y,
            cv=self.cv,
            scoring=self.scoring,
        )

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Log to MLflow
        if log_to_mlflow and self.use_mlflow and MLFLOW_AVAILABLE:
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    "n_features": len(features),
                    "features": ",".join(features),
                    "model_type": type(self.estimator).__name__,
                })
                mlflow.log_metrics({
                    f"cv_{self.scoring}_mean": mean_score,
                    f"cv_{self.scoring}_std": std_score,
                })

        return mean_score

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform by selecting features.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with selected features only

        Example:
            >>> X_selected = selector.transform(X)
        """
        if not self.is_fitted_:
            raise ValueError("Selector not fitted")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            DataFrame with selected features

        Example:
            >>> X_selected = selector.fit_transform(X, y)
        """
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """
        Get list of selected features.

        Returns:
            List of feature names

        Example:
            >>> features = selector.get_selected_features()
            >>> print(f"Selected {len(features)} features")
        """
        if not self.is_fitted_:
            raise ValueError("Selector not fitted")

        return self.selected_features_

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on selection order/scores.

        Returns:
            DataFrame with features and importance scores

        Example:
            >>> importance = selector.get_feature_importance()
            >>> print(importance.head())
        """
        if not self.is_fitted_:
            raise ValueError("Selector not fitted")

        if self.feature_scores_ is None:
            return pd.DataFrame()

        rows = []
        for features, score in self.feature_scores_.items():
            rows.append({
                "features": ",".join(features),
                "n_features": len(features),
                "score": score,
            })

        return pd.DataFrame(rows).sort_values("score", ascending=False)


class ForwardSelector(BaseFeatureSelector):
    """
    Forward feature selection.

    Starts with no features and adds one feature at a time that
    improves the model most.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> selector = ForwardSelector(
        ...     estimator=LogisticRegression(),
        ...     max_features=10,
        ...     use_mlflow=True
        ... )
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        max_features: Optional[int] = None,
        min_improvement: float = 0.001,
        **kwargs
    ) -> None:
        """
        Initialize forward selector.

        Args:
            estimator: Model to use
            max_features: Maximum features to select (None = all)
            min_improvement: Minimum improvement to continue
            **kwargs: Arguments for BaseFeatureSelector
        """
        super().__init__(estimator, **kwargs)
        self.max_features = max_features
        self.min_improvement = min_improvement

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForwardSelector":
        """
        Fit forward feature selector.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Self

        Example:
            >>> selector.fit(X_train, y_train)
        """
        self.logger.info(
            "Starting forward feature selection",
            n_features=X.shape[1],
            max_features=self.max_features,
            scoring=self.scoring,
        )

        all_features = list(X.columns)
        selected = []
        remaining = all_features.copy()

        self.feature_scores_ = {}
        best_score = 0.0

        max_iter = self.max_features if self.max_features else len(all_features)

        for iteration in range(max_iter):
            scores = {}

            # Try adding each remaining feature
            for feature in remaining:
                candidate = selected + [feature]
                score = self._evaluate_features(X, y, candidate)
                scores[feature] = score

            # Get best feature to add
            best_feature = max(scores.items(), key=lambda x: x[1])
            feature_name, score = best_feature

            # Check improvement
            improvement = score - best_score

            if improvement < self.min_improvement:
                self.logger.info(
                    f"Stopping: improvement {improvement:.4f} < {self.min_improvement}"
                )
                break

            # Add feature
            selected.append(feature_name)
            remaining.remove(feature_name)
            best_score = score

            self.feature_scores_[tuple(selected)] = best_score

            self.logger.info(
                f"Iteration {iteration + 1}: Added '{feature_name}' "
                f"(score={score:.4f}, improvement={improvement:.4f})"
            )

            if len(remaining) == 0:
                break

        self.selected_features_ = selected
        self.best_score_ = best_score
        self.is_fitted_ = True

        self.logger.info(
            f"Forward selection completed: {len(selected)} features selected, "
            f"best score={best_score:.4f}"
        )

        return self


class BackwardSelector(BaseFeatureSelector):
    """
    Backward feature elimination.

    Starts with all features and removes one feature at a time that
    hurts the model least.

    Example:
        >>> selector = BackwardSelector(
        ...     estimator=LogisticRegression(),
        ...     min_features=5
        ... )
        >>> selector.fit(X_train, y_train)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        min_features: int = 1,
        max_pvalue: float = 0.05,
        **kwargs
    ) -> None:
        """
        Initialize backward selector.

        Args:
            estimator: Model to use
            min_features: Minimum features to keep
            max_pvalue: Maximum p-value to keep feature (if applicable)
            **kwargs: Arguments for BaseFeatureSelector
        """
        super().__init__(estimator, **kwargs)
        self.min_features = min_features
        self.max_pvalue = max_pvalue

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BackwardSelector":
        """
        Fit backward feature selector.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Self

        Example:
            >>> selector.fit(X_train, y_train)
        """
        self.logger.info(
            "Starting backward feature elimination",
            n_features=X.shape[1],
            min_features=self.min_features,
        )

        selected = list(X.columns)
        self.feature_scores_ = {}

        # Initial score
        best_score = self._evaluate_features(X, y, selected)
        self.feature_scores_[tuple(selected)] = best_score

        iteration = 0

        while len(selected) > self.min_features:
            scores = {}

            # Try removing each feature
            for feature in selected:
                candidate = [f for f in selected if f != feature]
                score = self._evaluate_features(X, y, candidate)
                scores[feature] = score

            # Find feature whose removal hurts least (highest score after removal)
            worst_feature = max(scores.items(), key=lambda x: x[1])
            feature_name, score_after_removal = worst_feature

            # Remove feature
            selected.remove(feature_name)
            self.feature_scores_[tuple(selected)] = score_after_removal

            self.logger.info(
                f"Iteration {iteration + 1}: Removed '{feature_name}' "
                f"(score after removal={score_after_removal:.4f})"
            )

            iteration += 1

        # Find best subset
        best_subset = max(self.feature_scores_.items(), key=lambda x: x[1])
        self.selected_features_ = list(best_subset[0])
        self.best_score_ = best_subset[1]
        self.is_fitted_ = True

        self.logger.info(
            f"Backward elimination completed: {len(self.selected_features_)} features selected, "
            f"best score={self.best_score_:.4f}"
        )

        return self


class StepwiseSelector(BaseFeatureSelector):
    """
    Stepwise feature selection (bidirectional).

    Combines forward and backward selection. At each step, can either
    add a feature (forward) or remove a feature (backward).

    Example:
        >>> selector = StepwiseSelector(
        ...     estimator=LogisticRegression(),
        ...     max_features=10,
        ...     use_mlflow=True
        ... )
        >>> selector.fit(X_train, y_train)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        max_features: Optional[int] = None,
        min_improvement: float = 0.001,
        **kwargs
    ) -> None:
        """
        Initialize stepwise selector.

        Args:
            estimator: Model to use
            max_features: Maximum features to select
            min_improvement: Minimum improvement to continue
            **kwargs: Arguments for BaseFeatureSelector
        """
        super().__init__(estimator, **kwargs)
        self.max_features = max_features
        self.min_improvement = min_improvement

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StepwiseSelector":
        """
        Fit stepwise feature selector.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Self

        Example:
            >>> selector.fit(X_train, y_train)
        """
        self.logger.info(
            "Starting stepwise feature selection",
            n_features=X.shape[1],
            max_features=self.max_features,
        )

        all_features = list(X.columns)
        selected = []
        remaining = all_features.copy()

        self.feature_scores_ = {}
        best_score = 0.0

        max_iter = self.max_features if self.max_features else len(all_features)
        iteration = 0

        while iteration < max_iter and len(remaining) > 0:
            improved = False

            # Try forward step (add feature)
            forward_scores = {}
            for feature in remaining:
                candidate = selected + [feature]
                score = self._evaluate_features(X, y, candidate)
                forward_scores[feature] = score

            best_forward = max(forward_scores.items(), key=lambda x: x[1]) if forward_scores else (None, 0)

            # Try backward step (remove feature) if we have features
            backward_scores = {}
            if len(selected) > 0:
                for feature in selected:
                    candidate = [f for f in selected if f != feature]
                    score = self._evaluate_features(X, y, candidate)
                    backward_scores[feature] = score

            best_backward = max(backward_scores.items(), key=lambda x: x[1]) if backward_scores else (None, 0)

            # Choose best action
            forward_improvement = best_forward[1] - best_score if best_forward[0] else -float('inf')
            backward_improvement = best_backward[1] - best_score if best_backward[0] else -float('inf')

            if forward_improvement > backward_improvement and forward_improvement > self.min_improvement:
                # Add feature
                feature_name, score = best_forward
                selected.append(feature_name)
                remaining.remove(feature_name)
                best_score = score
                improved = True

                self.logger.info(
                    f"Iteration {iteration + 1}: ADDED '{feature_name}' (score={score:.4f})"
                )

            elif backward_improvement > self.min_improvement:
                # Remove feature
                feature_name, score = best_backward
                selected.remove(feature_name)
                remaining.append(feature_name)
                best_score = score
                improved = True

                self.logger.info(
                    f"Iteration {iteration + 1}: REMOVED '{feature_name}' (score={score:.4f})"
                )

            if not improved:
                self.logger.info("Stopping: no improvement possible")
                break

            self.feature_scores_[tuple(selected)] = best_score
            iteration += 1

        self.selected_features_ = selected
        self.best_score_ = best_score
        self.is_fitted_ = True

        self.logger.info(
            f"Stepwise selection completed: {len(selected)} features selected, "
            f"best score={best_score:.4f}"
        )

        return self


class ExhaustiveSelector(BaseFeatureSelector):
    """
    Exhaustive feature selection.

    Tries all possible feature combinations and selects the best.

    WARNING: Computationally expensive! With N features, evaluates 2^N - 1 models.
    Only use with small feature sets (N <= 15).

    Example:
        >>> # Only use with small feature sets!
        >>> selector = ExhaustiveSelector(
        ...     estimator=LogisticRegression(),
        ...     max_features=5
        ... )
        >>> selector.fit(X_train[top_10_features], y_train)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        min_features: int = 1,
        max_features: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize exhaustive selector.

        Args:
            estimator: Model to use
            min_features: Minimum features in subset
            max_features: Maximum features in subset
            **kwargs: Arguments for BaseFeatureSelector
        """
        super().__init__(estimator, **kwargs)
        self.min_features = min_features
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExhaustiveSelector":
        """
        Fit exhaustive feature selector.

        Args:
            X: Feature matrix (should be small, <= 15 features)
            y: Target variable

        Returns:
            Self

        Example:
            >>> selector.fit(X_train, y_train)
        """
        n_features = X.shape[1]

        if n_features > 15:
            warnings.warn(
                f"Exhaustive search with {n_features} features will evaluate "
                f"{2**n_features - 1} models. This may take a very long time! "
                f"Consider using Forward/Backward/Stepwise selection instead.",
                UserWarning
            )

        self.logger.info(
            "Starting exhaustive feature selection",
            n_features=n_features,
            warning="This may take a long time!",
        )

        all_features = list(X.columns)
        max_size = self.max_features if self.max_features else n_features

        self.feature_scores_ = {}
        best_score = 0.0
        best_features = []

        total_combinations = sum(
            len(list(combinations(all_features, r)))
            for r in range(self.min_features, max_size + 1)
        )

        self.logger.info(f"Will evaluate {total_combinations} feature combinations")

        evaluated = 0

        for size in range(self.min_features, max_size + 1):
            for feature_subset in combinations(all_features, size):
                features = list(feature_subset)
                score = self._evaluate_features(X, y, features)

                self.feature_scores_[tuple(features)] = score

                if score > best_score:
                    best_score = score
                    best_features = features

                evaluated += 1

                if evaluated % 100 == 0:
                    self.logger.info(
                        f"Progress: {evaluated}/{total_combinations} combinations evaluated"
                    )

        self.selected_features_ = best_features
        self.best_score_ = best_score
        self.is_fitted_ = True

        self.logger.info(
            f"Exhaustive selection completed: {len(best_features)} features selected, "
            f"best score={best_score:.4f}"
        )

        return self
