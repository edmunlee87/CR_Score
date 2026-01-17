"""Integration test for enhanced feature engineering."""

import sys
import traceback

def test_imports():
    """Test that all new classes can be imported."""
    try:
        from cr_score.features import (
            # Core
            FeatureRecipe,
            FeatureEngineeringConfig,
            PandasFeatureEngineer,
            AggregationType,
            TimeWindow,
            # New enums
            MissingStrategy,
            DivideByZeroPolicy,
            # Metadata
            FeatureRegistry,
            FeatureMetadata,
            # Enhanced
            TemporalTrendFeatures,
            CategoricalEncoder,
            FeatureValidator,
            DependencyGraph,
        )
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False


def test_temporal_features():
    """Test temporal trend features."""
    try:
        import pandas as pd
        import numpy as np
        from cr_score.features import TemporalTrendFeatures
        
        # Create sample data
        df = pd.DataFrame({
            'customer_id': ['A'] * 12,
            'date': pd.date_range('2023-01-01', periods=12, freq='M'),
            'balance': range(1000, 2200, 100),
        })
        
        trend = TemporalTrendFeatures()
        df = trend.delta(df, 'balance', time_col='date', group_cols=['customer_id'])
        df = trend.momentum(df, 'balance', time_col='date', group_cols=['customer_id'], window=3)
        
        assert 'balance_delta' in df.columns
        assert 'balance_momentum_3' in df.columns
        
        print("[OK] Temporal features working")
        return True
    except Exception as e:
        print(f"[FAIL] Temporal features failed: {e}")
        traceback.print_exc()
        return False


def test_categorical_encoding():
    """Test categorical encoding."""
    try:
        import pandas as pd
        from cr_score.features import CategoricalEncoder
        
        df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B', 'C'] * 10,
            'target': [1, 0, 1, 1, 0, 0] * 10,
        })
        
        encoder = CategoricalEncoder()
        df = encoder.freq_encoding(df, 'category')
        df = encoder.target_mean_encoding(df, 'category', 'target')
        
        assert 'category_freq' in df.columns
        assert 'category_target_mean' in df.columns
        
        print("[OK] Categorical encoding working")
        return True
    except Exception as e:
        print(f"[FAIL] Categorical encoding failed: {e}")
        traceback.print_exc()
        return False


def test_feature_validation():
    """Test feature validation."""
    try:
        import pandas as pd
        import numpy as np
        from cr_score.features import FeatureValidator
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        
        validator = FeatureValidator()
        results = validator.validate_features(df)
        
        assert 'feature_1' in results
        assert 'missing_rate' in results['feature_1']
        
        print("[OK] Feature validation working")
        return True
    except Exception as e:
        print(f"[FAIL] Feature validation failed: {e}")
        traceback.print_exc()
        return False


def test_dependency_graph():
    """Test dependency graph."""
    try:
        from cr_score.features import DependencyGraph
        
        graph = DependencyGraph()
        graph.add_feature('a', [])
        graph.add_feature('b', ['a'])
        graph.add_feature('c', ['b'])
        
        order = graph.topological_sort()
        
        assert order.index('a') < order.index('b')
        assert order.index('b') < order.index('c')
        
        print("[OK] Dependency graph working")
        return True
    except Exception as e:
        print(f"[FAIL] Dependency graph failed: {e}")
        traceback.print_exc()
        return False


def test_feature_registry():
    """Test feature registry."""
    try:
        from cr_score.features import FeatureRegistry
        
        registry = FeatureRegistry()
        registry.register(
            name='test_feature',
            source_columns=['col1'],
            operation='max',
            parameters={},
            window=None,
            missing_strategy='keep',
            dependencies=[],
            engine='pandas',
            output_dtype='float64'
        )
        
        assert 'test_feature' in registry.features
        
        print("[OK] Feature registry working")
        return True
    except Exception as e:
        print(f"[FAIL] Feature registry failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test full integration."""
    try:
        import pandas as pd
        import numpy as np
        from cr_score.features import (
            FeatureRecipe,
            FeatureEngineeringConfig,
            PandasFeatureEngineer,
            AggregationType,
            TimeWindow,
            TemporalTrendFeatures,
            CategoricalEncoder,
        )
        
        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame({
            'customer_id': ['A'] * 12 + ['B'] * 12,
            'date': pd.date_range('2023-01-01', periods=12, freq='M').tolist() * 2,
            'balance': np.random.randint(1000, 5000, 24),
            'credit_limit': np.random.randint(5000, 10000, 24),
            'category': np.random.choice(['X', 'Y', 'Z'], 24),
        })
        
        # 1. Apply temporal features
        trend = TemporalTrendFeatures()
        df = trend.delta(df, 'balance', time_col='date', group_cols=['customer_id'])
        
        # 2. Apply categorical encoding
        encoder = CategoricalEncoder()
        df = encoder.freq_encoding(df, 'category')
        
        # 3. Apply core feature engineering
        recipes = [
            FeatureRecipe("max_balance", "balance", AggregationType.MAX),
            FeatureRecipe("utilization", ["balance", "credit_limit"], "ratio"),
        ]
        
        config = FeatureEngineeringConfig(
            recipes=recipes,
            group_cols=["customer_id"]
        )
        
        engineer = PandasFeatureEngineer(config)
        df = engineer.fit_transform(df)
        
        # Verify all features created
        assert 'balance_delta' in df.columns
        assert 'category_freq' in df.columns
        assert 'max_balance' in df.columns
        assert 'utilization' in df.columns
        
        print("[OK] Full integration working")
        print(f"  Created {len(df.columns)} total columns")
        return True
    except Exception as e:
        print(f"[FAIL] Integration failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*80)
    print("Enhanced Feature Engineering - Integration Test")
    print("="*80)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Temporal Features", test_temporal_features),
        ("Categorical Encoding", test_categorical_encoding),
        ("Feature Validation", test_feature_validation),
        ("Dependency Graph", test_dependency_graph),
        ("Feature Registry", test_feature_registry),
        ("Full Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        success = test_func()
        results.append(success)
        print()
    
    print("="*80)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*80)
    
    if all(results):
        print("\nSUCCESS: All integration tests PASSED!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some tests FAILED")
        sys.exit(1)
