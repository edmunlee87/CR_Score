# CR_Score

**Enterprise Scorecard Development Platform**

Config-driven, Spark-native platform for end-to-end credit scorecard development with deterministic reproducibility and enterprise-grade audit trails.

## Key Features

- **Config-First Design**: Every action expressible as YAML configuration
- **Artifact-First**: All outputs versioned, hashed, and auditable
- **Deterministic by Default**: Same config + same data = same result
- **Spark-Native**: Efficient processing of 100M+ rows with intelligent compression
- **Scale Without Losing Correctness**: Sample weighting preserves likelihoods exactly
- **Multiple Interfaces**: CLI, SDK, API, UI
- **MCP/Tool Integration**: Agent-ready workflows
- **Enterprise Audit Trails**: Structured logging for compliance

## Architecture

```
Data → EDA → Feature Engineering → Binning → WoE Encoding →  
Reject Inference → Modeling → Calibration → Scaling → Reporting → Export
```

### Core Principles

1. **Config-First**: No hardcoded values, all defaults overridable
2. **Artifact-First**: Every step produces versioned artifacts
3. **Deterministic**: Reproducible results with fixed seeds and hashing
4. **Spark Where It Matters**: Heavy operations in Spark, orchestration in Python
5. **Scale Without Losing Correctness**: Post-binning compression reduces data 20x-100x

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/edmunlee87/CR_Score.git
cd CR_Score

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e ".[dev]"
```

### Basic Usage

**CLI Interface:**

```bash
# Validate configuration
cr-score validate --config config.yml

# Run scorecard development
cr-score run --config config.yml

# List runs
cr-score list-runs --limit 10

# Compare runs for reproducibility
cr-score compare --run-id-a run_123 --run-id-b run_456
```

**Python SDK (Complete Workflow):**

```python
# See examples/complete_scorecard_workflow.py for full example

from cr_score.binning import FineClasser, CoarseClasser
from cr_score.encoding import WoEEncoder
from cr_score.model import LogisticScorecard
from cr_score.scaling import PDOScaler

# 1. Binning
classer = FineClasser(method="quantile", max_bins=10)
classer.fit(df["age"], df["target"])
df["age_bin"] = classer.transform(df["age"])

# 2. WoE Encoding
encoder = WoEEncoder()
encoder.fit(df["age_bin"], df["target"])
df["age_woe"] = encoder.transform(df["age_bin"])
print(f"IV: {encoder.get_iv():.3f}")

# 3. Modeling
model = LogisticScorecard()
model.fit(X_woe, y, sample_weight=weights)
predictions = model.predict_proba(X_test_woe)[:, 1]

# 4. Scaling
scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)
scores = scaler.transform(predictions)
print(f"Mean score: {scores.mean():.0f}")
```

Run the complete example:

```bash
python examples/complete_scorecard_workflow.py
```

### Configuration Example

```yaml
project:
  name: "retail_scorecard"
  owner: "risk_team"
  
execution:
  engine: "spark_local"
  
data:
  sources:
    - path: "data/applications.parquet"
      format: "parquet"
      
target:
  definition: "default_flag"
  horizon_months: 12
  
binning:
  fine:
    method: "quantile"
    max_bins: 20
  coarse:
    monotonicity: true
    
model:
  type: "logistic"
  
scaling:
  pdo: 20
  base_score: 600
  base_odds: 50.0
```

See `src/cr_score/templates/intermediate/config_template.yml` for full example.

## Project Structure

```
CR_Score/
├── src/cr_score/
│   ├── core/              # Config, registry, logging, hashing
│   ├── data/              # Connectors, validation, optimization
│   ├── spark/             # Spark session, compression, metrics
│   ├── eda/               # Exploratory data analysis (pending)
│   ├── binning/           # Binning engine (pending)
│   ├── encoding/          # WoE encoding (pending)
│   ├── model/             # Modeling (pending)
│   ├── scaling/           # Score scaling (pending)
│   ├── reporting/         # Report generation (pending)
│   ├── cli/               # Command-line interface
│   └── templates/         # Configuration templates
├── tests/                 # Test suite
├── docs/                  # Documentation
└── pyproject.toml         # Package configuration
```

## Current Status

### ✅ Completed (v0.2.0-beta) - 67% Complete

**Core Infrastructure** (100% Complete)
- ✅ Config system with Pydantic validation (all URD schemas)
- ✅ Artifact registry & hashing for reproducibility
- ✅ Structured audit logging with JSON output
- ✅ CLI interface (validate, run, compare, list-runs)

**Data Layer** (100% Complete)
- ✅ Local file connectors (CSV, Parquet, JSON, Excel, Feather)
- ✅ Schema validation & data quality checks
- ✅ Column pruning & type optimization

**Spark Layer** (100% Complete)
- ✅ Spark session factory with config-driven setup
- ✅ Post-binning exact compression with sample weighting (20x-100x reduction)
- ✅ Verification with 0.0 tolerance for correctness

**EDA Module** (100% Complete)
- ✅ Univariate analysis (numeric/categorical statistics)
- ✅ Bivariate analysis (correlations, chi-square, Cramér's V)
- ✅ Drift analysis (PSI/CSI calculation)

**Binning Engine** (100% Complete)
- ✅ Fine classing (quantile, equal-width, decision tree)
- ✅ Coarse classing with monotonicity enforcement
- ✅ Monotonic merge algorithm

**WoE Encoding** (100% Complete)
- ✅ Weight of Evidence calculation
- ✅ Information Value (IV) with interpretation
- ✅ Multi-feature batch encoding

**Reject Inference** (100% Complete)
- ✅ Parceling method (score-based assignment)
- ✅ Reweighting method (propensity-based)

**Modeling** (100% Complete)
- ✅ Logistic regression with sample weighting
- ✅ Comprehensive diagnostics (AUC, Gini, KS, ROC)
- ✅ Model export and coefficient extraction

**Calibration & Scaling** (100% Complete)
- ✅ Intercept calibration for target bad rates
- ✅ PDO (Points-Double-Odds) transformation
- ✅ Score band generation
- ✅ Bidirectional score/probability conversion

### 🚧 In Progress (33% Remaining)

- Reporting and visualization modules
- MCP tools and permissions system
- Comprehensive test suite
- CI/CD pipeline
- Sphinx documentation

## Development

### Requirements

- Python 3.9+
- PySpark 3.4+
- pandas 2.0+
- Other dependencies in `pyproject.toml`

### Running Tests

```bash
pytest tests/ -v --cov=cr_score
```

### Code Quality

```bash
# Type checking
mypy src/cr_score

# Linting
ruff check src/cr_score

# Formatting
black src/cr_score
```

## Documentation

- **URD (User Requirements Document)**: `requirement/URD_v1.2.txt`
- **Agent Rules**: `docs/rules/AGENT_RULES.md`
- **Coding Standards**: `docs/rules/CODING_STANDARDS.md`
- **Validation Gates**: `docs/rules/VALIDATION_GATES.md`
- **Spark Operations Guide**: `docs/rules/SPARK_OPERATIONS_GUIDE.md`
- **Permissions Matrix**: `docs/rules/PERMISSIONS_MATRIX.md`
- **Artifact Specification**: `docs/rules/ARTIFACT_SPECIFICATION.md`

## License

Proprietary

## Contact

- **Author**: Edmun Lee
- **GitHub**: https://github.com/edmunlee87/CR_Score

## Roadmap

### v0.2.0 (Q2 2026)
- Complete EDA module
- Binning engine with monotonic merge
- WoE encoding

### v0.3.0 (Q3 2026)
- Modeling and calibration
- Score scaling
- Reporting

### v1.0.0 (Q4 2026)
- MCP tools
- API and UI
- Complete test suite
- Production deployment guide

## Citation

If you use CR_Score in your work, please cite:

```
CR_Score: Enterprise Scorecard Development Platform
Version 1.2.0
https://github.com/edmunlee87/CR_Score
```
