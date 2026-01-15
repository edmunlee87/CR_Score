"""
Complete Scorecard Development Workflow Example

Demonstrates the full CR_Score pipeline from raw data to credit scores.
"""

import pandas as pd
import numpy as np

# Import CR_Score modules
from cr_score.data.connectors import LocalConnector
from cr_score.data.validation import SchemaValidator, DataQualityChecker
from cr_score.data.optimization import ColumnPruner, TypeOptimizer
from cr_score.eda import UnivariateAnalyzer, BivariateAnalyzer, DriftAnalyzer
from cr_score.binning import FineClasser, CoarseClasser
from cr_score.encoding import WoEEncoder
from cr_score.model import LogisticScorecard
from cr_score.calibration import InterceptCalibrator
from cr_score.scaling import PDOScaler


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
    """Run complete scorecard development workflow."""
    print("=" * 80)
    print("CR_Score: Complete Scorecard Development Workflow")
    print("=" * 80)

    # Step 1: Load Data
    print("\n[1/10] Loading data...")
    df = generate_sample_data(n_samples=10000)
    print(f"Loaded {len(df)} samples")

    # Step 2: Data Quality Checks
    print("\n[2/10] Running data quality checks...")
    dq_checker = DataQualityChecker()
    dq_report = dq_checker.run_checks(df, target_col="default", id_col="customer_id")
    print(f"Critical issues: {dq_report['summary']['critical_issues']}")
    print(f"Warnings: {dq_report['summary']['warnings']}")

    # Step 3: Data Optimization
    print("\n[3/10] Optimizing data...")
    pruner = ColumnPruner()
    df_pruned = pruner.prune(
        df,
        target_col="default",
        feature_cols=["age", "income", "credit_score", "debt_ratio", "num_accounts"],
        segment_cols=["product"],
        id_col="customer_id",
    )

    optimizer = TypeOptimizer()
    df_opt, opt_report = optimizer.optimize(df_pruned)
    print(f"Memory savings: {opt_report['memory_savings_pct']:.1f}%")

    # Step 4: EDA
    print("\n[4/10] Running exploratory data analysis...")
    uni_analyzer = UnivariateAnalyzer()
    uni_stats = uni_analyzer.analyze(df_opt, target_col="default")

    bi_analyzer = BivariateAnalyzer()
    bi_results = bi_analyzer.analyze(df_opt, target_col="default")
    print(f"Top predictive features:")
    for _, row in bi_results.sort_values("correlation_abs", ascending=False).head(3).iterrows():
        print(f"  {row['feature']}: correlation={row['correlation']:.3f}")

    # Step 5: Train/Test Split
    print("\n[5/10] Splitting data...")
    train_idx = df_opt.index[:7000]
    test_idx = df_opt.index[7000:]

    df_train = df_opt.loc[train_idx]
    df_test = df_opt.loc[test_idx]
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # Step 6: Binning
    print("\n[6/10] Creating bins...")
    features_to_bin = ["age", "income", "credit_score", "debt_ratio", "num_accounts"]

    df_train_binned = df_train.copy()
    df_test_binned = df_test.copy()

    for feature in features_to_bin:
        # Fine binning
        fine_classer = FineClasser(method="quantile", max_bins=10)
        fine_classer.fit(df_train[feature], df_train["default"])

        df_train_binned[f"{feature}_bin"] = fine_classer.transform(df_train[feature])
        df_test_binned[f"{feature}_bin"] = fine_classer.transform(df_test[feature])

        print(f"  {feature}: {len(fine_classer.bin_labels_)} bins")

    # Step 7: WoE Encoding
    print("\n[7/10] Calculating WoE...")
    woe_features = [f"{f}_bin" for f in features_to_bin]

    df_train_woe = pd.DataFrame(index=df_train.index)
    df_test_woe = pd.DataFrame(index=df_test.index)

    for feature in woe_features:
        encoder = WoEEncoder()
        encoder.fit(df_train_binned[feature], df_train["default"])

        df_train_woe[f"{feature}_woe"] = encoder.transform(df_train_binned[feature])
        df_test_woe[f"{feature}_woe"] = encoder.transform(df_test_binned[feature])

        iv = encoder.get_iv()
        print(f"  {feature}: IV={iv:.3f} ({encoder.interpret_iv()})")

    # Step 8: Model Training
    print("\n[8/10] Training logistic regression...")
    model = LogisticScorecard()
    model.fit(df_train_woe, df_train["default"])

    # Get predictions
    train_proba = model.predict_proba(df_train_woe)[:, 1]
    test_proba = model.predict_proba(df_test_woe)[:, 1]

    # Evaluate
    train_metrics = model.get_performance_metrics(df_train["default"], train_proba)
    test_metrics = model.get_performance_metrics(df_test["default"], test_proba)

    print(f"  Train AUC: {train_metrics['auc']:.3f}, Gini: {train_metrics['gini']:.3f}")
    print(f"  Test  AUC: {test_metrics['auc']:.3f}, Gini: {test_metrics['gini']:.3f}")

    # Step 9: Calibration
    print("\n[9/10] Calibrating scores...")
    calibrator = InterceptCalibrator(target_bad_rate=0.05)
    calibrator.fit(test_proba, df_test["default"])

    test_proba_calibrated = calibrator.transform(test_proba)
    print(f"  Target bad rate: {calibrator.target_bad_rate:.2%}")
    print(f"  Achieved bad rate: {test_proba_calibrated.mean():.2%}")

    # Step 10: Scaling
    print("\n[10/10] Scaling to credit scores...")
    scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

    test_scores = scaler.transform(test_proba_calibrated)

    print(f"  Score range: {test_scores.min():.0f} - {test_scores.max():.0f}")
    print(f"  Mean score: {test_scores.mean():.0f}")
    print(f"\n  Score interpretation:")
    print(f"    Score 600 = 50:1 odds = 2% default rate")
    print(f"    Score 620 = 100:1 odds = 1% default rate")
    print(f"    Score 580 = 25:1 odds = 4% default rate")

    # Generate score bands
    score_bands = scaler.create_score_bands(n_bands=5)
    print(f"\n  Score bands:")
    for _, band in score_bands.iterrows():
        print(f"    Band {band['band']}: {band['score_min']}-{band['score_max']} "
              f"(Default rate: {band['default_probability']:.2%})")

    print("\n" + "=" * 80)
    print("Scorecard development complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
