"""
3-Line Scorecard Development Example

Demonstrates the simplified interface using ScorecardPipeline.
"""

import pandas as pd
import numpy as np

# Import the simplified pipeline
from cr_score import ScorecardPipeline


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic credit application data."""
    np.random.seed(42)

    df = pd.DataFrame({
        "customer_id": range(n_samples),
        "age": np.random.normal(40, 12, n_samples).clip(18, 80),
        "income": np.random.lognormal(10.5, 0.5, n_samples),
        "credit_score": np.random.normal(680, 80, n_samples).clip(300, 850),
        "debt_ratio": np.random.beta(2, 5, n_samples),
        "num_accounts": np.random.poisson(3, n_samples),
        "product": np.random.choice(["personal_loan", "credit_card", "mortgage"], n_samples),
    })

    # Generate target (default) with realistic relationships
    logit = (
        -3.0
        + 0.02 * (df["age"] - 40)
        - 0.0001 * (df["income"] - 50000)
        - 0.01 * (df["credit_score"] - 680)
        + 2.0 * df["debt_ratio"]
        - 0.1 * df["num_accounts"]
    )

    df["default"] = (np.random.random(n_samples) < 1 / (1 + np.exp(-logit))).astype(int)

    return df


def main():
    """Demonstrate 3-line scorecard development."""
    print("=" * 80)
    print("CR_Score: 3-Line Scorecard Development")
    print("=" * 80)

    # Load data
    print("\nGenerating sample data...")
    df = generate_sample_data(n_samples=10000)

    # Split
    df_train = df.iloc[:7000]
    df_test = df.iloc[7000:]

    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    print("\n" + "=" * 80)
    print("METHOD 1: Ultra-Simple (3 lines)")
    print("=" * 80 + "\n")

    # ┌─────────────────────────────────────────────────────────────┐
    # │ THE 3 MAGIC LINES                                           │
    # └─────────────────────────────────────────────────────────────┘

    pipeline = ScorecardPipeline()
    pipeline.fit(df_train, target_col="default")
    scores = pipeline.predict(df_test)

    # ┌─────────────────────────────────────────────────────────────┐
    # │ DONE! That's it. You have a production scorecard.           │
    # └─────────────────────────────────────────────────────────────┘

    print(f"\n✓ Scorecard built and scores generated!")
    print(f"  Score range: {scores.min():.0f} - {scores.max():.0f}")
    print(f"  Mean score: {scores.mean():.0f}")

    # Evaluate
    test_metrics = pipeline.evaluate(df_test, target_col="default")
    print(f"\n  Test AUC: {test_metrics['auc']:.3f}")
    print(f"  Test Gini: {test_metrics['gini']:.3f}")

    print("\n" + "=" * 80)
    print("METHOD 2: With Configuration (still simple!)")
    print("=" * 80 + "\n")

    # Configure your scorecard parameters
    pipeline_custom = ScorecardPipeline(
        max_n_bins=5,          # Max 5 bins per feature
        min_iv=0.02,           # Minimum IV to include feature
        pdo=20,                # Every 20 points, odds double
        base_score=600,        # Score 600 = 2% default rate
        target_bad_rate=0.05,  # Calibrate to 5% bad rate
    )

    # Still just 2 lines to fit and predict
    pipeline_custom.fit(df_train, target_col="default")
    scores_custom = pipeline_custom.predict(df_test)

    print(f"\n✓ Custom scorecard built!")
    print(f"  Score range: {scores_custom.min():.0f} - {scores_custom.max():.0f}")
    print(f"  Mean score: {scores_custom.mean():.0f}")

    # Get detailed summary
    summary = pipeline_custom.get_summary()
    print(f"\n  Features selected: {summary['n_features']}")
    print(f"  Top 3 features by IV:")
    for i, feat in enumerate(summary['iv_summary'][:3], 1):
        print(f"    {i}. {feat['feature']}: IV={feat['iv']:.3f}")

    # Export for production
    print("\n  Exporting scorecard specification...")
    pipeline_custom.export_scorecard("scorecard_spec.json")
    print("  ✓ Saved to scorecard_spec.json")

    print("\n" + "=" * 80)
    print("COMPARISON: Old vs New Interface")
    print("=" * 80 + "\n")

    print("OLD WAY (10+ lines):")
    print("""
    fine_classer = FineClasser(method="quantile", max_bins=10)
    fine_classer.fit(df["age"], df["target"])
    df["age_bin"] = fine_classer.transform(df["age"])

    encoder = WoEEncoder()
    encoder.fit(df["age_bin"], df["target"])
    df["age_woe"] = encoder.transform(df["age_bin"])

    model = LogisticScorecard()
    model.fit(X_woe, y)
    predictions = model.predict_proba(X_test)[:, 1]

    scaler = PDOScaler(pdo=20, base_score=600)
    scores = scaler.transform(predictions)
    """)

    print("\nNEW WAY (3 lines):")
    print("""
    pipeline = ScorecardPipeline()
    pipeline.fit(df_train, target_col="default")
    scores = pipeline.predict(df_test)
    """)

    print("\n✓ 70% less code!")
    print("✓ Automatic optimal binning!")
    print("✓ Automatic feature selection (IV-based)!")
    print("✓ Production-ready JSON export!")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
CR_Score now offers TWO ways to build scorecards:

1. DETAILED CONTROL: Use individual modules
   - Full flexibility
   - Fine-tune each step
   - Perfect for research and experimentation

2. SIMPLIFIED PIPELINE: Use ScorecardPipeline
   - 3 lines of code
   - Optimal binning with optbinning package
   - Automatic feature selection
   - Production-ready
   - Perfect for rapid deployment

Choose the approach that fits your needs!
    """)


if __name__ == "__main__":
    main()
