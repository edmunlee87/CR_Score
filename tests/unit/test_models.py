"""
Unit tests for all model families.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from cr_score.model import (
    BaseScorecardModel,
    LogisticScorecard,
    RandomForestScorecard,
)


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    # Split into train/test
    train_size = 700
    X_train = X_df.iloc[:train_size]
    X_test = X_df.iloc[train_size:]
    y_train = y_series.iloc[:train_size]
    y_test = y_series.iloc[train_size:]
    
    return X_train, X_test, y_train, y_test


class TestLogisticScorecard:
    """Tests for LogisticScorecard."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LogisticScorecard(regularization='l2', C=1.0, random_state=42)
        assert model.regularization == 'l2'
        assert model.C == 1.0
        assert model.random_state == 42
        assert not model.is_fitted_
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X_train, _, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted_
        assert model.feature_names_ == list(X_train.columns)
        assert model.model_ is not None
    
    def test_fit_with_sample_weights(self, sample_data):
        """Test model fitting with sample weights."""
        X_train, _, y_train, _ = sample_data
        weights = pd.Series(np.random.uniform(0.5, 2.0, len(y_train)))
        
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        
        assert model.is_fitted_
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_predict(self, sample_data):
        """Test class prediction."""
        X_train, X_test, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test, threshold=0.5)
        
        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})
    
    def test_get_coefficients(self, sample_data):
        """Test coefficient retrieval."""
        X_train, _, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        coefs = model.get_coefficients()
        
        assert isinstance(coefs, pd.DataFrame)
        assert len(coefs) == len(X_train.columns)
        assert 'feature' in coefs.columns
        assert 'coefficient' in coefs.columns
        assert 'abs_coefficient' in coefs.columns
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance."""
        X_train, _, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == len(X_train.columns)
    
    def test_get_performance_metrics(self, sample_data):
        """Test comprehensive metrics."""
        X_train, X_test, y_train, y_test = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)[:, 1]
        metrics = model.get_performance_metrics(y_test, probas)
        
        assert 'classification' in metrics
        assert 'ranking' in metrics
        assert 'calibration' in metrics
        assert 'interpretations' in metrics
        
        # Check key metrics exist
        assert 'auc' in metrics['ranking']
        assert 'precision' in metrics['classification']
    
    def test_export_model(self, sample_data):
        """Test model export."""
        X_train, _, y_train, _ = sample_data
        model = LogisticScorecard(random_state=42)
        model.fit(X_train, y_train)
        
        exported = model.export_model()
        
        assert exported['model_type'] == 'LogisticScorecard'
        assert exported['feature_names'] == list(X_train.columns)
        assert exported['random_state'] == 42
        assert exported['is_fitted']


class TestRandomForestScorecard:
    """Tests for RandomForestScorecard."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = RandomForestScorecard(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        assert model.n_estimators == 50
        assert model.max_depth == 3
        assert model.random_state == 42
        assert not model.is_fitted_
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        X_train, _, y_train, _ = sample_data
        model = RandomForestScorecard(n_estimators=10, random_state=42)
        
        model.fit(X_train, y_train)
        
        assert model.is_fitted_
        assert model.feature_names_ == list(X_train.columns)
    
    def test_predict_proba(self, sample_data):
        """Test probability prediction."""
        X_train, X_test, y_train, _ = sample_data
        model = RandomForestScorecard(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.all((probas >= 0) & (probas <= 1))
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance."""
        X_train, _, y_train, _ = sample_data
        model = RandomForestScorecard(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == len(X_train.columns)
    
    def test_get_tree_depths(self, sample_data):
        """Test tree depth statistics."""
        X_train, _, y_train, _ = sample_data
        model = RandomForestScorecard(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        depths = model.get_tree_depths()
        
        assert 'min' in depths
        assert 'max' in depths
        assert 'mean' in depths
        assert 'median' in depths
        assert depths['max'] <= 5


class TestXGBoostScorecard:
    """Tests for XGBoostScorecard (optional dependency)."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", minversion=None),
        reason="XGBoost not installed"
    )
    def test_initialization(self):
        """Test model initialization."""
        from cr_score.model import XGBoostScorecard
        
        model = XGBoostScorecard(n_estimators=50, max_depth=3, random_state=42)
        assert model.n_estimators == 50
        assert model.max_depth == 3
    
    @pytest.mark.skipif(
        not pytest.importorskip("xgboost", minversion=None),
        reason="XGBoost not installed"
    )
    def test_fit_and_predict(self, sample_data):
        """Test training and prediction."""
        from cr_score.model import XGBoostScorecard
        
        X_train, X_test, y_train, _ = sample_data
        model = XGBoostScorecard(n_estimators=10, random_state=42)
        
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        
        assert model.is_fitted_
        assert probas.shape == (len(X_test), 2)


class TestLightGBMScorecard:
    """Tests for LightGBMScorecard (optional dependency)."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", minversion=None),
        reason="LightGBM not installed"
    )
    def test_initialization(self):
        """Test model initialization."""
        from cr_score.model import LightGBMScorecard
        
        model = LightGBMScorecard(n_estimators=50, num_leaves=31, random_state=42)
        assert model.n_estimators == 50
        assert model.num_leaves == 31
    
    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", minversion=None),
        reason="LightGBM not installed"
    )
    def test_fit_and_predict(self, sample_data):
        """Test training and prediction."""
        from cr_score.model import LightGBMScorecard
        
        X_train, X_test, y_train, _ = sample_data
        model = LightGBMScorecard(n_estimators=10, random_state=42)
        
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        
        assert model.is_fitted_
        assert probas.shape == (len(X_test), 2)


def test_sklearn_compatibility(sample_data):
    """Test sklearn compatibility (clone, cross_val_score)."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    
    X_train, _, y_train, _ = sample_data
    
    # Test clone
    model = LogisticScorecard(random_state=42)
    cloned = clone(model)
    assert cloned.random_state == 42
    assert not cloned.is_fitted_
    
    # Test cross_val_score
    scores = cross_val_score(
        LogisticScorecard(random_state=42),
        X_train,
        y_train,
        cv=3,
        scoring='roc_auc'
    )
    assert len(scores) == 3
    assert all(score > 0.5 for score in scores)
