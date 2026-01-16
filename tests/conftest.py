"""
Pytest configuration and shared fixtures.

Provides common test fixtures and setup for all tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_binary_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)

    feature_names = [f"feat_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def sample_credit_df():
    """Generate sample credit dataset."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "age": np.random.randint(18, 75, n),
        "income": np.random.randint(20000, 200000, n),
        "debt_ratio": np.random.uniform(0, 1, n),
        "credit_history": np.random.randint(0, 30, n),
        "num_accounts": np.random.randint(1, 10, n),
    })

    # Generate target
    logit = (
        -2
        + 0.02 * (df["age"] - 40)
        + 0.000005 * (df["income"] - 60000)
        - 1.5 * df["debt_ratio"]
        + 0.03 * df["credit_history"]
    )

    prob = 1 / (1 + np.exp(-logit))
    df["default"] = (np.random.random(n) < prob).astype(int)

    return df


@pytest.fixture
def sample_binning_table():
    """Generate sample binning table."""
    return pd.DataFrame({
        "bin": ["[0, 30)", "[30, 40)", "[40, 50)", "[50, 100)"],
        "total_count": [250, 300, 300, 150],
        "event_count": [50, 45, 30, 10],
        "event_rate": [0.20, 0.15, 0.10, 0.067],
        "woe": [0.693, 0.287, -0.105, -0.693],
        "iv_contribution": [0.049, 0.015, 0.003, 0.035],
    })
