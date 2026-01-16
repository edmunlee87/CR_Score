"""
Tests for ScorecardPipeline.

Tests the complete scorecard development pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from cr_score import ScorecardPipeline


@pytest.fixture
def sample_credit_data():
    """Generate sample credit data for testing."""
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.randint(20000, 150000, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "debt_ratio": np.random.uniform(0, 1, n_samples),
        "employment_years": np.random.randint(0, 40, n_samples),
    })

    # Generate target based on features
    logit = (
        -3.0
        + 0.03 * (df["age"] - 40)
        + 0.00001 * (df["income"] - 70000)
        + 0.005 * (df["credit_score"] - 600)
        - 2.0 * df["debt_ratio"]
        + 0.02 * df["employment_years"]
    )

    prob = 1 / (1 + np.exp(-logit))
    df["default"] = (np.random.random(n_samples) < prob).astype(int)

    return df


def test_pipeline_basic_fit(sample_credit_data):
    """Test basic pipeline fitting."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        max_n_bins=5,
        pdo=20,
        base_score=600,
        random_state=42,
    )

    pipeline.fit(df, target_col="default")

    assert pipeline.is_fitted_
    assert pipeline.auto_binner_ is not None
    assert pipeline.model_ is not None
    assert pipeline.scaler_ is not None


def test_pipeline_predict(sample_credit_data):
    """Test pipeline prediction."""
    df = sample_credit_data
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    pipeline = ScorecardPipeline(
        max_n_bins=5,
        pdo=20,
        base_score=600,
        random_state=42,
    )

    pipeline.fit(df_train, target_col="default")
    scores = pipeline.predict(df_test)

    assert len(scores) == len(df_test)
    assert scores.min() > 0
    assert scores.max() < 1000


def test_pipeline_predict_proba(sample_credit_data):
    """Test pipeline probability prediction."""
    df = sample_credit_data
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    pipeline = ScorecardPipeline(
        max_n_bins=5,
        random_state=42,
    )

    pipeline.fit(df_train, target_col="default")
    probas = pipeline.predict_proba(df_test)

    assert len(probas) == len(df_test)
    assert (probas >= 0).all() and (probas <= 1).all()


def test_pipeline_evaluate(sample_credit_data):
    """Test pipeline evaluation."""
    df = sample_credit_data
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    pipeline = ScorecardPipeline(random_state=42)

    pipeline.fit(df_train, target_col="default")
    metrics = pipeline.evaluate(df_test, target_col="default")

    assert "auc" in metrics
    assert "gini" in metrics
    assert "ks" in metrics
    assert metrics["auc"] > 0.5  # Better than random


def test_pipeline_with_feature_selection(sample_credit_data):
    """Test pipeline with feature selection."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        max_n_bins=5,
        feature_selection="forward",
        max_features=3,
        random_state=42,
    )

    pipeline.fit(df, target_col="default")

    assert pipeline.is_fitted_
    assert pipeline.feature_selector_ is not None
    assert len(pipeline.selected_features_) <= 3


def test_pipeline_summary(sample_credit_data):
    """Test pipeline summary generation."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(random_state=42)
    pipeline.fit(df, target_col="default")

    summary = pipeline.get_summary()

    assert "n_features" in summary
    assert "selected_features" in summary
    assert "iv_summary" in summary
    assert "model_coefficients" in summary
    assert "pdo_params" in summary


def test_pipeline_without_calibration(sample_credit_data):
    """Test pipeline without calibration."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        calibrate=False,
        random_state=42,
    )

    pipeline.fit(df, target_col="default")

    assert pipeline.is_fitted_
    assert pipeline.calibrator_ is None


def test_pipeline_with_custom_pdo(sample_credit_data):
    """Test pipeline with custom PDO parameters."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        pdo=50,
        base_score=700,
        base_odds=30.0,
        random_state=42,
    )

    pipeline.fit(df, target_col="default")
    scores = pipeline.predict(df)

    # Check that scores are in reasonable range for these params
    assert scores.mean() > 500
    assert scores.mean() < 900


def test_pipeline_not_fitted_error(sample_credit_data):
    """Test that using unfitted pipeline raises error."""
    df = sample_credit_data

    pipeline = ScorecardPipeline()

    with pytest.raises(ValueError, match="not fitted"):
        pipeline.predict(df)

    with pytest.raises(ValueError, match="not fitted"):
        pipeline.predict_proba(df)

    with pytest.raises(ValueError, match="not fitted"):
        pipeline.get_summary()


def test_pipeline_stepwise_selection(sample_credit_data):
    """Test pipeline with stepwise selection."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        feature_selection="stepwise",
        max_features=3,
        random_state=42,
    )

    pipeline.fit(df, target_col="default")

    assert pipeline.is_fitted_
    assert pipeline.feature_selector_ is not None
    assert len(pipeline.selected_features_) <= 3


def test_pipeline_backward_selection(sample_credit_data):
    """Test pipeline with backward selection."""
    df = sample_credit_data

    pipeline = ScorecardPipeline(
        feature_selection="backward",
        random_state=42,
    )

    pipeline.fit(df, target_col="default")

    assert pipeline.is_fitted_
    assert pipeline.feature_selector_ is not None
