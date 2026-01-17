# Feature Engineering Toolkit

Comprehensive, configurable feature engineering toolkit for credit risk modeling.

## Overview

The feature engineering toolkit provides a powerful, config-driven approach to creating derived features for credit scorecard development. It supports both single feature creation and batch mode operations, works with pandas and PySpark, and includes a rich set of aggregation and transformation operations.

## Key Features

- **🔧 Configurable**: Define features via Python configs or YAML files
- **⚡ Batch & Single Mode**: Create features one-by-one or in batches
- **🐼 Pandas & Spark**: Works with both pandas and PySpark DataFrames
- **📊 Rich Operations**: 10+ aggregation types + transformations
- **⏰ Time Windows**: Built-in support for time-based aggregations
- **🎯 Credit Risk Focused**: Designed for scorecard features

## Quick Start

### Single Feature Creation

```python
from cr_score.features import PandasFeatureEngineer, AggregationType

# Initialize engineer
engineer = PandasFeatureEngineer()

# Create max delinquency feature
df = engineer.create_aggregation(
    df,
    feature_name="max_dpd_3m",
    source_col="days_past_due",
    operation=AggregationType.MAX,
    group_by="customer_id",
    window_months=3,
    time_col="date"
)

# Create utilization ratio
df = engineer.create_ratio(
    df,
    feature_name="utilization",
    numerator_col="balance",
    denominator_col="credit_limit"
)
```

### Batch Mode with Configuration

```python
from cr_score.features import (
    FeatureRecipe,
    FeatureEngineeringConfig,
    PandasFeatureEngineer,
    AggregationType,
    TimeWindow,
)

# Define feature recipes
recipes = [
    FeatureRecipe(
        name="max_dpd_3m",
        source_cols="days_past_due",
        operation=AggregationType.MAX,
        window=TimeWindow.LAST_3M,
        description="Max DPD in last 3 months"
    ),
    FeatureRecipe(
        name="utilization",
        source_cols=["balance", "credit_limit"],
        operation="ratio",
        description="Credit utilization"
    ),
    FeatureRecipe(
        name="avg_payment",
        source_cols="payment_amount",
        operation=AggregationType.MEAN,
        description="Average payment"
    ),
]

# Create configuration
config = FeatureEngineeringConfig(
    recipes=recipes,
    id_col="customer_id",
    time_col="snapshot_date",
    group_cols=["customer_id"]
)

# Apply all features
engineer = PandasFeatureEngineer(config)
df_transformed = engineer.fit_transform(df)

print(f"Created {len(engineer.created_features_)} features")
```

### Load Configuration from YAML

```python
import yaml
from cr_score.features import FeatureEngineeringConfig, create_feature_engineer

# Load config from YAML
with open("feature_config.yml") as f:
    config_dict = yaml.safe_load(f)

config = FeatureEngineeringConfig.from_dict(config_dict)

# Create engineer and apply
engineer = create_feature_engineer(config, engine="pandas")
df_transformed = engineer.fit_transform(df)
```

## Available Operations

### Aggregation Types

| Operation | Description | Example Use Case |
|-----------|-------------|------------------|
| `MAX` | Maximum value | Max delinquency |
| `MIN` | Minimum value | Min balance |
| `MEAN` | Average value | Avg payment amount |
| `MEDIAN` | Median value | Median utilization |
| `STD` | Standard deviation | Payment volatility |
| `SUM` | Sum of values | Total payments |
| `COUNT` | Count of records | Number of transactions |
| `FIRST` | First value | Opening balance |
| `LAST` | Last value | Current balance |
| `RANGE` | Max - Min | Balance range |
| `WORST` | Same as MAX | Worst delinquency |

### Transformation Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `ratio` | A / B | Requires 2 source columns |
| `difference` | A - B | Requires 2 source columns |
| `product` | A × B × ... | Multiple source columns |
| `log` | log(x) or log1p(x) | `add_one`: bool (default True) |
| `sqrt` | √x | None |
| `clip` | Clip to range | `lower`, `upper` |
| `bin` | Binning/discretization | `bins`, `labels` |

### Time Windows

| Window | Description |
|--------|-------------|
| `LAST_1M` | Last 1 month |
| `LAST_3M` | Last 3 months |
| `LAST_6M` | Last 6 months |
| `LAST_12M` | Last 12 months |
| `LAST_24M` | Last 24 months |
| `ALL_TIME` | All available data |

## Examples

### Example 1: Delinquency Features

```python
recipes = [
    FeatureRecipe("max_dpd_3m", "dpd", AggregationType.MAX, TimeWindow.LAST_3M),
    FeatureRecipe("max_dpd_6m", "dpd", AggregationType.MAX, TimeWindow.LAST_6M),
    FeatureRecipe("max_dpd_12m", "dpd", AggregationType.MAX, TimeWindow.LAST_12M),
    FeatureRecipe("avg_dpd_12m", "dpd", AggregationType.MEAN, TimeWindow.LAST_12M),
]
```

### Example 2: Balance & Utilization Features

```python
recipes = [
    FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
    FeatureRecipe("balance_volatility", "balance", AggregationType.STD),
    FeatureRecipe("balance_range", "balance", AggregationType.RANGE),
    FeatureRecipe("utilization", ["balance", "credit_limit"], "ratio"),
    FeatureRecipe("available_credit", ["credit_limit", "balance"], "difference"),
]
```

### Example 3: Payment Behavior Features

```python
recipes = [
    FeatureRecipe("total_payments_12m", "payment", AggregationType.SUM, TimeWindow.LAST_12M),
    FeatureRecipe("avg_payment", "payment", AggregationType.MEAN),
    FeatureRecipe("payment_consistency", "payment", AggregationType.STD),
]
```

### Example 4: Transformation Features

```python
recipes = [
    FeatureRecipe("log_balance", "balance", "log", params={"add_one": True}),
    FeatureRecipe("sqrt_debt", "total_debt", "sqrt"),
    FeatureRecipe("dpd_capped", "dpd", "clip", params={"lower": 0, "upper": 90}),
]
```

### Example 5: Rolling Window Features

```python
engineer = PandasFeatureEngineer()

# 3-month rolling average balance
df = engineer.create_rolling_feature(
    df,
    feature_name="balance_ma3",
    source_col="balance",
    window=3,
    operation=AggregationType.MEAN,
    group_by="customer_id"
)

# 6-month rolling max delinquency
df = engineer.create_rolling_feature(
    df,
    feature_name="max_dpd_6m_rolling",
    source_col="days_past_due",
    window=6,
    operation=AggregationType.MAX,
    group_by="customer_id"
)
```

## Using with PySpark

```python
from cr_score.features import SparkFeatureEngineer

# Create config (same as pandas)
config = FeatureEngineeringConfig(
    recipes=[...],
    group_cols=["customer_id"]
)

# Use Spark engineer
engineer = SparkFeatureEngineer(config)
spark_df_transformed = engineer.fit_transform(spark_df)

# Or use factory
engineer = create_feature_engineer(config, engine="spark")
```

## YAML Configuration Format

```yaml
# feature_config.yml
id_col: customer_id
time_col: snapshot_date
group_cols:
  - customer_id

recipes:
  - name: max_dpd_3m
    source_cols: days_past_due
    operation: max
    window: last_3_months
    description: "Maximum DPD in last 3 months"
  
  - name: utilization
    source_cols:
      - balance
      - credit_limit
    operation: ratio
    description: "Credit utilization ratio"
  
  - name: log_debt
    source_cols: total_debt
    operation: log
    params:
      add_one: true
    description: "Log-transformed debt"
```

## API Reference

### Classes

#### `FeatureRecipe`
Configuration for a single feature.

**Parameters:**
- `name` (str): Output feature name
- `source_cols` (str | List[str]): Input column(s)
- `operation` (AggregationType | str): Operation to apply
- `window` (TimeWindow | str, optional): Time window for aggregation
- `params` (dict, optional): Additional parameters
- `description` (str, optional): Human-readable description

#### `FeatureEngineeringConfig`
Batch configuration for multiple features.

**Parameters:**
- `recipes` (List[FeatureRecipe]): List of feature recipes
- `id_col` (str, optional): ID column name
- `time_col` (str, optional): Time column name
- `group_cols` (List[str], optional): Columns to group by

#### `PandasFeatureEngineer`
Feature engineer for pandas DataFrames.

**Methods:**
- `fit_transform(df)`: Apply all configured features
- `transform(df)`: Apply features (no fitting needed)
- `create_aggregation(...)`: Create single aggregation feature
- `create_ratio(...)`: Create single ratio feature
- `create_rolling_feature(...)`: Create rolling window feature

#### `SparkFeatureEngineer`
Feature engineer for PySpark DataFrames.

**Methods:**
- `fit_transform(df)`: Apply all configured features
- `transform(df)`: Apply features (no fitting needed)
- `create_aggregation(...)`: Create single aggregation feature

### Functions

#### `create_feature_engineer(config, engine="pandas")`
Factory function to create feature engineer.

**Parameters:**
- `config` (FeatureEngineeringConfig, optional): Configuration
- `engine` (str): "pandas" or "spark"

**Returns:**
- Feature engineer instance

## Common Patterns

### Pattern 1: Typical Credit Scorecard Features

```python
recipes = [
    # Delinquency
    FeatureRecipe("max_dpd_3m", "dpd", AggregationType.MAX, TimeWindow.LAST_3M),
    FeatureRecipe("max_dpd_6m", "dpd", AggregationType.MAX, TimeWindow.LAST_6M),
    FeatureRecipe("max_dpd_12m", "dpd", AggregationType.MAX, TimeWindow.LAST_12M),
    
    # Balance & Utilization
    FeatureRecipe("avg_balance", "balance", AggregationType.MEAN),
    FeatureRecipe("utilization", ["balance", "limit"], "ratio"),
    
    # Payment
    FeatureRecipe("total_payments_12m", "payment", AggregationType.SUM, TimeWindow.LAST_12M),
    FeatureRecipe("payment_consistency", "payment", AggregationType.STD),
    
    # Account
    FeatureRecipe("debt_per_account", ["total_debt", "num_accounts"], "ratio"),
]
```

### Pattern 2: Time Series Features

```python
# Multiple time windows for same metric
for window in [TimeWindow.LAST_3M, TimeWindow.LAST_6M, TimeWindow.LAST_12M]:
    recipes.append(
        FeatureRecipe(
            f"max_dpd_{window.value}",
            "days_past_due",
            AggregationType.MAX,
            window
        )
    )
```

### Pattern 3: Derived Features Chain

```python
# Step 1: Create base features
df = engineer.create_ratio(df, "utilization", "balance", "limit")

# Step 2: Transform
recipes = [FeatureRecipe("log_util", "utilization", "log", params={"add_one": True})]
config = FeatureEngineeringConfig(recipes=recipes)
engineer2 = PandasFeatureEngineer(config)
df = engineer2.fit_transform(df)
```

## Best Practices

1. **Use descriptive names**: `max_dpd_3m` is better than `feat_1`
2. **Document with descriptions**: Add descriptions to recipes
3. **Group related features**: Organize recipes by category
4. **Test incrementally**: Start with a few features, then expand
5. **Use YAML for production**: Store configs in version control
6. **Handle missing data**: The toolkit handles NaN/null gracefully
7. **Monitor created features**: Check `engineer.created_features_`

## Performance Tips

### For Pandas
- Use batch mode instead of individual feature creation
- Consider using Spark for large datasets (>1M rows)
- Use appropriate data types (int32 vs int64)

### For Spark
- Cache intermediate results when creating many features
- Use appropriate number of partitions
- Consider broadcasting small lookup tables
- Use column pruning after feature creation

## Integration with CR-Score Pipeline

```python
from cr_score.features import create_feature_engineer, FeatureEngineeringConfig
from cr_score.encoding import WOEEncoder
from cr_score.model import LogisticScorecard

# 1. Feature engineering
config = FeatureEngineeringConfig.from_dict(yaml_config)
engineer = create_feature_engineer(config, engine="pandas")
df = engineer.fit_transform(df_raw)

# 2. WOE encoding
encoder = WOEEncoder()
df_woe = encoder.fit_transform(df[engineer.created_features_], y)

# 3. Model training
model = LogisticScorecard()
model.fit(df_woe, y)
```

## Examples

See `examples/feature_engineering_examples.py` for comprehensive examples including:
- Single feature creation
- Batch mode configuration
- Time-windowed features
- Mathematical transformations
- Rolling window features
- Configuration from YAML
- Complete credit scorecard feature set

## Testing

Run tests:
```bash
pytest tests/unit/test_feature_engineering.py -v
```

## Future Enhancements

- [ ] Categorical feature engineering
- [ ] Interaction terms
- [ ] Polynomial features
- [ ] Target encoding
- [ ] Automated feature selection integration
- [ ] Feature importance tracking
- [ ] Performance profiling

## Contributing

When adding new operations:
1. Add to `AggregationType` enum or as string operation
2. Implement in both `PandasFeatureEngineer` and `SparkFeatureEngineer`
3. Add tests in `test_feature_engineering.py`
4. Update this documentation

## License

Proprietary - CR-Score Platform
