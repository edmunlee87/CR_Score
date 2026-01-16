"""
Tests for feature selection module.

Tests all four selection methods with different estimators.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from cr_score.features import (
    ForwardSelector,
    BackwardSelector,
    StepwiseSelector,
    ExhaustiveSelector,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    
    # Only first 3 features are informative
    y = (
        0.5 * X[:, 0]
        - 0.3 * X[:, 1]
        + 0.4 * X[:, 2]
        + np.random.randn(n_samples) * 0.3
    )
    
    y = (y > np.median(y)).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def test_forward_selector_basic(sample_data):
    """Test forward selector with basic parameters."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=5,
        use_mlflow=False,
    )

    selector.fit(X, y)

    assert selector.is_fitted_
    assert len(selector.get_selected_features()) <= 5
    assert len(selector.get_selected_features()) > 0
    assert selector.best_score_ > 0


def test_backward_selector_basic(sample_data):
    """Test backward selector with basic parameters."""
    X, y = sample_data

    # Use smaller subset for faster test
    X_small = X.iloc[:, :5]

    selector = BackwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        min_features=2,
        use_mlflow=False,
    )

    selector.fit(X_small, y)

    assert selector.is_fitted_
    assert len(selector.get_selected_features()) >= 2
    assert selector.best_score_ > 0


def test_stepwise_selector_basic(sample_data):
    """Test stepwise selector with basic parameters."""
    X, y = sample_data

    selector = StepwiseSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=5,
        use_mlflow=False,
    )

    selector.fit(X, y)

    assert selector.is_fitted_
    assert len(selector.get_selected_features()) <= 5
    assert selector.best_score_ > 0


def test_exhaustive_selector_small(sample_data):
    """Test exhaustive selector with small feature set."""
    X, y = sample_data

    # Only use 5 features for exhaustive search
    X_small = X.iloc[:, :5]

    selector = ExhaustiveSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        min_features=2,
        max_features=3,
        use_mlflow=False,
    )

    selector.fit(X_small, y)

    assert selector.is_fitted_
    assert 2 <= len(selector.get_selected_features()) <= 3
    assert selector.best_score_ > 0


def test_model_agnostic_logistic(sample_data):
    """Test that selector works with logistic regression."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=3,
        use_mlflow=False,
    )

    selector.fit(X, y)
    assert selector.is_fitted_


def test_model_agnostic_random_forest(sample_data):
    """Test that selector works with random forest."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=RandomForestClassifier(n_estimators=10, random_state=42),
        max_features=3,
        use_mlflow=False,
    )

    selector.fit(X, y)
    assert selector.is_fitted_


def test_transform(sample_data):
    """Test transform method."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=3,
        use_mlflow=False,
    )

    selector.fit(X, y)
    X_transformed = selector.transform(X)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == len(selector.get_selected_features())
    assert list(X_transformed.columns) == selector.get_selected_features()


def test_fit_transform(sample_data):
    """Test fit_transform method."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=3,
        use_mlflow=False,
    )

    X_transformed = selector.fit_transform(X, y)

    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] <= 3
    assert selector.is_fitted_


def test_get_feature_importance(sample_data):
    """Test get_feature_importance method."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=3,
        use_mlflow=False,
    )

    selector.fit(X, y)
    importance = selector.get_feature_importance()

    assert isinstance(importance, pd.DataFrame)
    assert len(importance) > 0
    assert "score" in importance.columns


def test_not_fitted_error(sample_data):
    """Test that using unfitted selector raises error."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        use_mlflow=False,
    )

    with pytest.raises(ValueError, match="not fitted"):
        selector.transform(X)

    with pytest.raises(ValueError, match="not fitted"):
        selector.get_selected_features()


def test_min_improvement_stopping(sample_data):
    """Test that min_improvement stops selection appropriately."""
    X, y = sample_data

    selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=10,
        min_improvement=0.5,  # Very high threshold
        use_mlflow=False,
    )

    selector.fit(X, y)

    # Should stop early due to high min_improvement
    assert len(selector.get_selected_features()) < 10
