"""
Feature Selection with MLflow Tracking Example

Demonstrates model-agnostic feature selection with experiment tracking.
"""

import pandas as pd
import numpy as np

# Import feature selectors
from cr_score.features import (
    ForwardSelector,
    BackwardSelector,
    StepwiseSelector,
    ExhaustiveSelector,
)

# Import different model types to show model-agnostic capability
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def generate_sample_data(n_samples: int = 5000, n_features: int = 20) -> tuple:
    """Generate synthetic data with some informative and some noise features."""
    np.random.seed(42)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Only first 5 features are informative
    y = (
        0.5 * X[:, 0]
        - 0.3 * X[:, 1]
        + 0.4 * X[:, 2]
        - 0.2 * X[:, 3]
        + 0.3 * X[:, 4]
        + np.random.randn(n_samples) * 0.5
    )

    y = (y > np.median(y)).astype(int)

    # Create DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def main():
    """Demonstrate feature selection with different methods and models."""
    print("=" * 80)
    print("Feature Selection with MLflow Tracking")
    print("=" * 80)

    # Generate data
    print("\nGenerating sample data...")
    X, y = generate_sample_data(n_samples=5000, n_features=20)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"(Note: Only first 5 features are truly informative)")

    # Split data
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    print("\n" + "=" * 80)
    print("EXAMPLE 1: Forward Selection with Logistic Regression")
    print("=" * 80 + "\n")

    forward_selector = ForwardSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=10,
        min_improvement=0.001,
        scoring="roc_auc",
        cv=3,
        use_mlflow=True,
        mlflow_experiment_name="Feature_Selection_Forward",
    )

    forward_selector.fit(X_train, y_train)

    selected_features = forward_selector.get_selected_features()
    print(f"\n✓ Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    print(f"\n✓ Best CV Score: {forward_selector.best_score_:.4f}")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Backward Elimination with Random Forest")
    print("=" * 80 + "\n")

    # Start with top 10 features for faster demo
    top_features = X_train.columns[:10].tolist()
    X_train_subset = X_train[top_features]

    backward_selector = BackwardSelector(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        min_features=3,
        scoring="roc_auc",
        cv=3,
        use_mlflow=True,
        mlflow_experiment_name="Feature_Selection_Backward",
    )

    backward_selector.fit(X_train_subset, y_train)

    selected_features = backward_selector.get_selected_features()
    print(f"\n✓ Selected {len(selected_features)} features from {len(top_features)}:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    print(f"\n✓ Best CV Score: {backward_selector.best_score_:.4f}")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Stepwise Selection with Logistic Regression")
    print("=" * 80 + "\n")

    stepwise_selector = StepwiseSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        max_features=8,
        min_improvement=0.001,
        scoring="roc_auc",
        cv=3,
        use_mlflow=True,
        mlflow_experiment_name="Feature_Selection_Stepwise",
    )

    stepwise_selector.fit(X_train, y_train)

    selected_features = stepwise_selector.get_selected_features()
    print(f"\n✓ Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    print(f"\n✓ Best CV Score: {stepwise_selector.best_score_:.4f}")

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Exhaustive Search (Small Subset)")
    print("=" * 80 + "\n")

    print("⚠️  Exhaustive search is computationally expensive!")
    print("Using only first 8 features for demonstration...\n")

    # Only use first 8 features for exhaustive search
    small_subset = X_train.columns[:8].tolist()
    X_train_small = X_train[small_subset]

    exhaustive_selector = ExhaustiveSelector(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        min_features=3,
        max_features=5,
        scoring="roc_auc",
        cv=3,
        use_mlflow=True,
        mlflow_experiment_name="Feature_Selection_Exhaustive",
    )

    exhaustive_selector.fit(X_train_small, y_train)

    selected_features = exhaustive_selector.get_selected_features()
    print(f"\n✓ Selected {len(selected_features)} features (globally optimal):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")

    print(f"\n✓ Best CV Score: {exhaustive_selector.best_score_:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON: Method Performance")
    print("=" * 80 + "\n")

    methods = [
        ("Forward Selection", forward_selector),
        ("Backward Elimination", backward_selector),
        ("Stepwise Selection", stepwise_selector),
        ("Exhaustive Search", exhaustive_selector),
    ]

    for method_name, selector in methods:
        n_features = len(selector.get_selected_features())
        score = selector.best_score_
        print(f"{method_name:25s}: {n_features:2d} features, CV Score = {score:.4f}")

    print("\n" + "=" * 80)
    print("MODEL-AGNOSTIC CAPABILITY")
    print("=" * 80 + "\n")

    print("These selectors work with ANY sklearn-compatible model:")
    print("""
Examples:
    - LogisticRegression()
    - RandomForestClassifier()
    - XGBClassifier()
    - LightGBMClassifier()
    - SVC()
    - GradientBoostingClassifier()
    - Any custom estimator with fit() and predict_proba()

Just pass any model to the selector:

    selector = ForwardSelector(
        estimator=YourFavoriteModel(),
        ...
    )
    """)

    print("\n" + "=" * 80)
    print("MLFLOW TRACKING")
    print("=" * 80 + "\n")

    print("All experiments are tracked in MLflow!")
    print("""
To view results:
    1. Run: mlflow ui
    2. Open: http://localhost:5000
    3. Browse experiments and compare feature selections

Each run logs:
    - Features used
    - CV scores (mean & std)
    - Model type
    - Number of features

This makes it easy to:
    ✓ Compare different selection methods
    ✓ Compare different models
    ✓ Track experiments over time
    ✓ Reproduce results
    """)

    print("\n" + "=" * 80)
    print("USAGE IN SCORECARD PIPELINE")
    print("=" * 80 + "\n")

    print("Integrate with scorecard development:")
    print("""
from cr_score.features import ForwardSelector
from cr_score.model import LogisticScorecard

# 1. Select features
selector = ForwardSelector(
    estimator=LogisticScorecard(),
    max_features=10
)
selector.fit(X_train_woe, y_train)

# 2. Get selected features
selected = selector.get_selected_features()

# 3. Train final model with selected features only
X_train_selected = X_train_woe[selected]
final_model = LogisticScorecard()
final_model.fit(X_train_selected, y_train)

# 4. Use for scoring
X_test_selected = X_test_woe[selected]
scores = final_model.predict_proba(X_test_selected)[:, 1]
    """)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
CR_Score now provides:

✓ Model-Agnostic Feature Selection
  - Works with ANY sklearn-compatible model
  - Four methods: Forward, Backward, Stepwise, Exhaustive
  - Cross-validation based evaluation

✓ MLflow Integration
  - Automatic experiment tracking
  - Easy comparison of methods
  - Reproducible results

✓ Production-Ready
  - Comprehensive logging
  - Configurable parameters
  - Fits into existing workflows

Choose the right method:
    - Forward:     Fast, good for many features
    - Backward:    Good when starting point is known
    - Stepwise:    Most flexible, best overall
    - Exhaustive:  Optimal but slow (only for small feature sets)
    """)


if __name__ == "__main__":
    main()
