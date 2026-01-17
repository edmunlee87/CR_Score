"""
Example: Feature Engineering Toolkit

This example demonstrates how to use the feature engineering toolkit for creating
derived features in credit risk modeling.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from cr_score.features import (
    AggregationType,
    TimeWindow,
    FeatureRecipe,
    FeatureEngineeringConfig,
    PandasFeatureEngineer,
    create_feature_engineer,
)


def create_sample_credit_data():
    """Create sample credit bureau data."""
    np.random.seed(42)
    
    # Create customer-month level data
    customers = [f"CUST_{i:04d}" for i in range(1, 101)]
    months = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    
    data = []
    for customer in customers:
        for month in months:
            data.append({
                'customer_id': customer,
                'snapshot_date': month,
                'balance': np.random.randint(500, 10000),
                'credit_limit': np.random.randint(5000, 20000),
                'days_past_due': np.random.choice([0, 0, 0, 0, 0, 0, 0, 15, 30, 60, 90], p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002]),
                'payment_amount': np.random.randint(100, 2000),
                'num_inquiries': np.random.randint(0, 5),
                'num_accounts': np.random.randint(1, 10),
                'total_debt': np.random.randint(1000, 50000),
            })
    
    return pd.DataFrame(data)


def example_1_single_features():
    """Example 1: Creating single features."""
    print("=" * 80)
    print("Example 1: Creating Single Features")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_credit_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize engineer
    engineer = PandasFeatureEngineer()
    
    # Create max delinquency per customer
    print("\n1. Creating max_dpd feature...")
    df = engineer.create_aggregation(
        df,
        feature_name="max_dpd",
        source_col="days_past_due",
        operation=AggregationType.MAX,
        group_by="customer_id"
    )
    print(f"   Created: max_dpd")
    
    # Create average balance per customer
    print("\n2. Creating avg_balance feature...")
    df = engineer.create_aggregation(
        df,
        feature_name="avg_balance",
        source_col="balance",
        operation=AggregationType.MEAN,
        group_by="customer_id"
    )
    print(f"   Created: avg_balance")
    
    # Create utilization ratio
    print("\n3. Creating utilization ratio...")
    df = engineer.create_ratio(
        df,
        feature_name="utilization",
        numerator_col="balance",
        denominator_col="credit_limit"
    )
    print(f"   Created: utilization")
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"\nSample of new features:")
    print(df[['customer_id', 'max_dpd', 'avg_balance', 'utilization']].head(10))
    
    return df


def example_2_batch_mode():
    """Example 2: Batch mode with configuration."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Mode with Configuration")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_credit_data()
    
    # Define feature recipes
    recipes = [
        # Delinquency features
        FeatureRecipe(
            name="max_dpd",
            source_cols="days_past_due",
            operation=AggregationType.MAX,
            description="Maximum delinquency"
        ),
        FeatureRecipe(
            name="avg_dpd",
            source_cols="days_past_due",
            operation=AggregationType.MEAN,
            description="Average delinquency"
        ),
        FeatureRecipe(
            name="worst_dpd",
            source_cols="days_past_due",
            operation=AggregationType.WORST,
            description="Worst delinquency (same as max)"
        ),
        
        # Balance features
        FeatureRecipe(
            name="avg_balance",
            source_cols="balance",
            operation=AggregationType.MEAN,
            description="Average balance"
        ),
        FeatureRecipe(
            name="balance_volatility",
            source_cols="balance",
            operation=AggregationType.STD,
            description="Balance volatility"
        ),
        FeatureRecipe(
            name="balance_range",
            source_cols="balance",
            operation=AggregationType.RANGE,
            description="Balance range (max - min)"
        ),
        
        # Utilization
        FeatureRecipe(
            name="utilization",
            source_cols=["balance", "credit_limit"],
            operation="ratio",
            description="Credit utilization"
        ),
        
        # Payment behavior
        FeatureRecipe(
            name="total_payments",
            source_cols="payment_amount",
            operation=AggregationType.SUM,
            description="Total payments"
        ),
        FeatureRecipe(
            name="avg_payment",
            source_cols="payment_amount",
            operation=AggregationType.MEAN,
            description="Average payment"
        ),
        
        # Account features
        FeatureRecipe(
            name="debt_per_account",
            source_cols=["total_debt", "num_accounts"],
            operation="ratio",
            description="Debt per account"
        ),
    ]
    
    # Create configuration
    config = FeatureEngineeringConfig(
        recipes=recipes,
        id_col="customer_id",
        group_cols=["customer_id"]
    )
    
    print(f"\nNumber of recipes: {len(recipes)}")
    print("\nRecipes:")
    for i, recipe in enumerate(recipes, 1):
        print(f"{i:2d}. {recipe.name:25s} = {recipe.operation:10s} ({recipe.description})")
    
    # Create engineer and transform
    engineer = PandasFeatureEngineer(config)
    df_transformed = engineer.fit_transform(df)
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"New features created: {len(engineer.created_features_)}")
    
    print("\nSample output:")
    feature_cols = ['customer_id'] + engineer.created_features_
    print(df_transformed[feature_cols].head(10))
    
    return df_transformed


def example_3_time_windows():
    """Example 3: Time-based features with windows."""
    print("\n" + "=" * 80)
    print("Example 3: Time-Based Features with Windows")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_credit_data()
    
    # Define recipes with time windows
    recipes = [
        FeatureRecipe(
            name="max_dpd_3m",
            source_cols="days_past_due",
            operation=AggregationType.MAX,
            window=TimeWindow.LAST_3M,
            description="Max DPD in last 3 months"
        ),
        FeatureRecipe(
            name="max_dpd_6m",
            source_cols="days_past_due",
            operation=AggregationType.MAX,
            window=TimeWindow.LAST_6M,
            description="Max DPD in last 6 months"
        ),
        FeatureRecipe(
            name="max_dpd_12m",
            source_cols="days_past_due",
            operation=AggregationType.MAX,
            window=TimeWindow.LAST_12M,
            description="Max DPD in last 12 months"
        ),
        FeatureRecipe(
            name="avg_balance_3m",
            source_cols="balance",
            operation=AggregationType.MEAN,
            window=TimeWindow.LAST_3M,
            description="Avg balance in last 3 months"
        ),
        FeatureRecipe(
            name="avg_balance_6m",
            source_cols="balance",
            operation=AggregationType.MEAN,
            window=TimeWindow.LAST_6M,
            description="Avg balance in last 6 months"
        ),
    ]
    
    config = FeatureEngineeringConfig(
        recipes=recipes,
        id_col="customer_id",
        time_col="snapshot_date",
        group_cols=["customer_id"]
    )
    
    print(f"\nCreating time-windowed features...")
    engineer = PandasFeatureEngineer(config)
    df_transformed = engineer.fit_transform(df)
    
    print(f"\nFeatures created: {engineer.created_features_}")
    
    # Show sample customer
    sample_customer = df_transformed['customer_id'].iloc[0]
    print(f"\nSample customer: {sample_customer}")
    print(df_transformed[df_transformed['customer_id'] == sample_customer][
        ['snapshot_date', 'max_dpd_3m', 'max_dpd_6m', 'max_dpd_12m', 'avg_balance_3m', 'avg_balance_6m']
    ].tail(5))
    
    return df_transformed


def example_4_transformations():
    """Example 4: Mathematical transformations."""
    print("\n" + "=" * 80)
    print("Example 4: Mathematical Transformations")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_credit_data()
    
    # Define transformation recipes
    recipes = [
        # Log transformations
        FeatureRecipe(
            name="log_balance",
            source_cols="balance",
            operation="log",
            params={"add_one": True},
            description="Log(balance + 1)"
        ),
        FeatureRecipe(
            name="log_debt",
            source_cols="total_debt",
            operation="log",
            params={"add_one": True},
            description="Log(total_debt + 1)"
        ),
        
        # Square root
        FeatureRecipe(
            name="sqrt_balance",
            source_cols="balance",
            operation="sqrt",
            description="Square root of balance"
        ),
        
        # Clipping
        FeatureRecipe(
            name="dpd_capped",
            source_cols="days_past_due",
            operation="clip",
            params={"lower": 0, "upper": 90},
            description="DPD capped at 90"
        ),
        
        # Difference
        FeatureRecipe(
            name="available_credit",
            source_cols=["credit_limit", "balance"],
            operation="difference",
            description="Available credit"
        ),
        
        # Product
        FeatureRecipe(
            name="total_inquiry_risk",
            source_cols=["num_inquiries", "days_past_due"],
            operation="product",
            description="Inquiry * DPD"
        ),
    ]
    
    config = FeatureEngineeringConfig(recipes=recipes)
    
    print(f"\nApplying {len(recipes)} transformations...")
    engineer = PandasFeatureEngineer(config)
    df_transformed = engineer.fit_transform(df)
    
    print(f"\nFeatures created: {engineer.created_features_}")
    
    print("\nSample transformations:")
    comparison_cols = [
        ('balance', 'log_balance', 'sqrt_balance'),
        ('days_past_due', 'dpd_capped', None),
        ('credit_limit', 'available_credit', None),
    ]
    
    sample = df_transformed.head(5)
    for original, *transformed in comparison_cols:
        print(f"\n{original}:")
        print(sample[[original] + [t for t in transformed if t]])
    
    return df_transformed


def example_5_rolling_windows():
    """Example 5: Rolling window features."""
    print("\n" + "=" * 80)
    print("Example 5: Rolling Window Features")
    print("=" * 80)
    
    # Create sample time series data for a single customer
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')
    df = pd.DataFrame({
        'customer_id': 'CUST_0001',
        'date': dates,
        'balance': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100],
        'dpd': [0, 0, 30, 0, 0, 0, 60, 0, 0, 0, 0, 0],
        'payment': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
    })
    
    print("\nOriginal time series:")
    print(df[['date', 'balance', 'dpd', 'payment']])
    
    # Create rolling features
    engineer = PandasFeatureEngineer()
    
    print("\nCreating rolling features...")
    
    # 3-month rolling average balance
    df = engineer.create_rolling_feature(
        df,
        feature_name="balance_ma3",
        source_col="balance",
        window=3,
        operation=AggregationType.MEAN,
        group_by="customer_id"
    )
    
    # 6-month rolling max DPD
    df = engineer.create_rolling_feature(
        df,
        feature_name="max_dpd_6m",
        source_col="dpd",
        window=6,
        operation=AggregationType.MAX,
        group_by="customer_id"
    )
    
    # 3-month rolling sum payments
    df = engineer.create_rolling_feature(
        df,
        feature_name="payment_sum_3m",
        source_col="payment",
        window=3,
        operation=AggregationType.SUM,
        group_by="customer_id"
    )
    
    print("\nTime series with rolling features:")
    print(df[['date', 'balance', 'balance_ma3', 'dpd', 'max_dpd_6m', 'payment', 'payment_sum_3m']])
    
    return df


def example_6_config_from_dict():
    """Example 6: Loading configuration from dictionary."""
    print("\n" + "=" * 80)
    print("Example 6: Configuration from Dictionary")
    print("=" * 80)
    
    # Configuration as dictionary (could be loaded from YAML/JSON)
    config_dict = {
        "recipes": [
            {
                "name": "max_dpd_3m",
                "source_cols": "days_past_due",
                "operation": "max",
                "window": "last_3_months",
                "description": "Max DPD in last 3 months"
            },
            {
                "name": "avg_balance_6m",
                "source_cols": "balance",
                "operation": "mean",
                "window": "last_6_months",
                "description": "Average balance in last 6 months"
            },
            {
                "name": "utilization",
                "source_cols": ["balance", "credit_limit"],
                "operation": "ratio",
                "description": "Credit utilization ratio"
            },
            {
                "name": "debt_per_account",
                "source_cols": ["total_debt", "num_accounts"],
                "operation": "ratio",
                "description": "Average debt per account"
            },
        ],
        "id_col": "customer_id",
        "time_col": "snapshot_date",
        "group_cols": ["customer_id"]
    }
    
    print("Configuration dictionary:")
    print(f"  - Number of recipes: {len(config_dict['recipes'])}")
    print(f"  - ID column: {config_dict['id_col']}")
    print(f"  - Time column: {config_dict['time_col']}")
    print(f"  - Group columns: {config_dict['group_cols']}")
    
    # Create config from dict
    config = FeatureEngineeringConfig.from_dict(config_dict)
    
    # Create sample data
    df = create_sample_credit_data()
    
    # Apply features
    engineer = PandasFeatureEngineer(config)
    df_transformed = engineer.fit_transform(df)
    
    print(f"\nFeatures created: {engineer.created_features_}")
    print(f"\nTransformed data shape: {df_transformed.shape}")
    
    print("\nSample output:")
    print(df_transformed[['customer_id', 'max_dpd_3m', 'avg_balance_6m', 'utilization', 'debt_per_account']].head(10))
    
    return df_transformed


def example_7_credit_scorecard():
    """Example 7: Complete credit scorecard feature set."""
    print("\n" + "=" * 80)
    print("Example 7: Complete Credit Scorecard Feature Set")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_credit_data()
    
    print("\nBuilding comprehensive feature set for credit scoring...")
    
    # Define comprehensive feature set
    recipes = [
        # ===== DELINQUENCY FEATURES =====
        FeatureRecipe("max_dpd_3m", "days_past_due", AggregationType.MAX, TimeWindow.LAST_3M,
                     description="Max DPD - Last 3 months"),
        FeatureRecipe("max_dpd_6m", "days_past_due", AggregationType.MAX, TimeWindow.LAST_6M,
                     description="Max DPD - Last 6 months"),
        FeatureRecipe("max_dpd_12m", "days_past_due", AggregationType.MAX, TimeWindow.LAST_12M,
                     description="Max DPD - Last 12 months"),
        FeatureRecipe("avg_dpd_12m", "days_past_due", AggregationType.MEAN, TimeWindow.LAST_12M,
                     description="Average DPD - Last 12 months"),
        
        # ===== BALANCE & UTILIZATION =====
        FeatureRecipe("avg_balance", "balance", AggregationType.MEAN,
                     description="Average balance"),
        FeatureRecipe("max_balance", "balance", AggregationType.MAX,
                     description="Maximum balance"),
        FeatureRecipe("balance_trend", "balance", AggregationType.RANGE,
                     description="Balance volatility (range)"),
        FeatureRecipe("utilization", ["balance", "credit_limit"], "ratio",
                     description="Credit utilization ratio"),
        FeatureRecipe("available_credit", ["credit_limit", "balance"], "difference",
                     description="Available credit"),
        
        # ===== PAYMENT BEHAVIOR =====
        FeatureRecipe("total_payments_12m", "payment_amount", AggregationType.SUM, TimeWindow.LAST_12M,
                     description="Total payments - Last 12 months"),
        FeatureRecipe("avg_payment_12m", "payment_amount", AggregationType.MEAN, TimeWindow.LAST_12M,
                     description="Average payment - Last 12 months"),
        FeatureRecipe("payment_consistency", "payment_amount", AggregationType.STD,
                     description="Payment consistency (std dev)"),
        
        # ===== ACCOUNT FEATURES =====
        FeatureRecipe("debt_per_account", ["total_debt", "num_accounts"], "ratio",
                     description="Debt per account"),
        FeatureRecipe("avg_num_accounts", "num_accounts", AggregationType.MEAN,
                     description="Average number of accounts"),
        
        # ===== INQUIRY FEATURES =====
        FeatureRecipe("total_inquiries_6m", "num_inquiries", AggregationType.SUM, TimeWindow.LAST_6M,
                     description="Total inquiries - Last 6 months"),
        FeatureRecipe("inquiry_rate", "num_inquiries", AggregationType.MEAN,
                     description="Average inquiry rate"),
        
        # ===== DERIVED FEATURES =====
        FeatureRecipe("log_debt", "total_debt", "log", params={"add_one": True},
                     description="Log-transformed total debt"),
        FeatureRecipe("sqrt_balance", "balance", "sqrt",
                     description="Square root of balance"),
    ]
    
    config = FeatureEngineeringConfig(
        recipes=recipes,
        id_col="customer_id",
        time_col="snapshot_date",
        group_cols=["customer_id"]
    )
    
    print(f"\nTotal features to create: {len(recipes)}")
    
    # Categorize features
    categories = {
        "Delinquency": [r for r in recipes if "dpd" in r.name.lower()],
        "Balance & Utilization": [r for r in recipes if any(x in r.name.lower() for x in ["balance", "util", "credit"])],
        "Payment Behavior": [r for r in recipes if "payment" in r.name.lower()],
        "Account Features": [r for r in recipes if "account" in r.name.lower()],
        "Inquiry Features": [r for r in recipes if "inquir" in r.name.lower()],
        "Derived Features": [r for r in recipes if any(x in r.name.lower() for x in ["log", "sqrt"])],
    }
    
    for category, category_recipes in categories.items():
        print(f"\n{category} ({len(category_recipes)} features):")
        for recipe in category_recipes:
            print(f"  - {recipe.name:30s}: {recipe.description}")
    
    # Apply all features
    print("\n" + "-" * 80)
    print("Applying feature engineering...")
    engineer = PandasFeatureEngineer(config)
    df_transformed = engineer.fit_transform(df)
    
    print(f"\nFeature engineering complete!")
    print(f"  Original columns: {df.shape[1]}")
    print(f"  New columns: {df_transformed.shape[1]}")
    print(f"  Features created: {len(engineer.created_features_)}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("Summary Statistics for Key Features:")
    print("-" * 80)
    
    key_features = ['max_dpd_3m', 'max_dpd_6m', 'utilization', 'avg_payment_12m', 'debt_per_account']
    print(df_transformed[key_features].describe())
    
    return df_transformed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CR-Score Feature Engineering Toolkit Examples")
    print("=" * 80)
    
    # Run all examples
    df1 = example_1_single_features()
    df2 = example_2_batch_mode()
    df3 = example_3_time_windows()
    df4 = example_4_transformations()
    df5 = example_5_rolling_windows()
    df6 = example_6_config_from_dict()
    df7 = example_7_credit_scorecard()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
