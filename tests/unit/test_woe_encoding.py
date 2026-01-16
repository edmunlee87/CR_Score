"""
Tests for WoE encoding module.

Tests Weight of Evidence calculation and Information Value.
"""

import numpy as np
import pandas as pd
import pytest

from cr_score.encoding.woe import WoEEncoder


@pytest.fixture
def simple_binned_data():
    """Generate simple binned data for testing."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "age_bin": np.random.choice(["[18-30)", "[30-50)", "[50-70)"], n),
        "income_bin": np.random.choice(["Low", "Medium", "High"], n),
        "default": np.random.randint(0, 2, n),
    })

    return df


def test_woe_encoder_fit(simple_binned_data):
    """Test WoE encoder fitting."""
    df = simple_binned_data

    encoder = WoEEncoder()
    encoder.fit(df, feature_cols=["age_bin", "income_bin"], target_col="default")

    assert encoder.is_fitted_
    assert len(encoder.woe_maps_) == 2
    assert "age_bin" in encoder.woe_maps_
    assert "income_bin" in encoder.woe_maps_


def test_woe_encoder_transform(simple_binned_data):
    """Test WoE encoding transformation."""
    df = simple_binned_data

    encoder = WoEEncoder()
    encoder.fit(df, feature_cols=["age_bin", "income_bin"], target_col="default")
    df_transformed = encoder.transform(df)

    assert "age_bin_woe" in df_transformed.columns
    assert "income_bin_woe" in df_transformed.columns
    assert len(df_transformed) == len(df)


def test_woe_encoder_fit_transform(simple_binned_data):
    """Test WoE fit_transform."""
    df = simple_binned_data

    encoder = WoEEncoder()
    df_transformed = encoder.fit_transform(
        df,
        feature_cols=["age_bin", "income_bin"],
        target_col="default"
    )

    assert encoder.is_fitted_
    assert "age_bin_woe" in df_transformed.columns
    assert "income_bin_woe" in df_transformed.columns


def test_woe_values_range(simple_binned_data):
    """Test that WoE values are in reasonable range."""
    df = simple_binned_data

    encoder = WoEEncoder()
    df_transformed = encoder.fit_transform(
        df,
        feature_cols=["age_bin"],
        target_col="default"
    )

    woe_values = df_transformed["age_bin_woe"]

    # WoE should typically be in range [-5, 5]
    assert woe_values.min() > -10
    assert woe_values.max() < 10


def test_iv_calculation(simple_binned_data):
    """Test Information Value calculation."""
    df = simple_binned_data

    encoder = WoEEncoder()
    encoder.fit(df, feature_cols=["age_bin", "income_bin"], target_col="default")

    iv_summary = encoder.get_iv_summary()

    assert isinstance(iv_summary, pd.DataFrame)
    assert len(iv_summary) == 2
    assert "feature" in iv_summary.columns
    assert "iv" in iv_summary.columns
    assert (iv_summary["iv"] >= 0).all()


def test_woe_table(simple_binned_data):
    """Test WoE table generation."""
    df = simple_binned_data

    encoder = WoEEncoder()
    encoder.fit(df, feature_cols=["age_bin"], target_col="default")

    woe_table = encoder.get_woe_table("age_bin")

    assert isinstance(woe_table, pd.DataFrame)
    assert "woe" in woe_table.columns
    assert "event_rate" in woe_table.columns
    assert len(woe_table) > 0


def test_not_fitted_error(simple_binned_data):
    """Test that using unfitted encoder raises error."""
    df = simple_binned_data

    encoder = WoEEncoder()

    with pytest.raises(ValueError, match="not fitted"):
        encoder.transform(df)

    with pytest.raises(ValueError, match="not fitted"):
        encoder.get_iv_summary()


def test_missing_feature_handling(simple_binned_data):
    """Test handling of missing features during transform."""
    df_train = simple_binned_data
    df_test = simple_binned_data.copy()

    encoder = WoEEncoder()
    encoder.fit(df_train, feature_cols=["age_bin"], target_col="default")

    # Transform should work even if test set has new values
    df_test.loc[0, "age_bin"] = "NEW_VALUE"
    df_transformed = encoder.transform(df_test)

    # Should handle gracefully (either default to 0 or mean)
    assert "age_bin_woe" in df_transformed.columns
    assert not df_transformed["age_bin_woe"].isna().all()


def test_get_selected_features(simple_binned_data):
    """Test get_selected_features method."""
    df = simple_binned_data

    encoder = WoEEncoder(min_iv=0.0)
    encoder.fit(df, feature_cols=["age_bin", "income_bin"], target_col="default")

    selected = encoder.get_selected_features()

    assert isinstance(selected, list)
    assert len(selected) >= 0
    assert len(selected) <= 2
