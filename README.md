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

```bash
# Validate configuration
cr-score validate --config config.yml

# Run scorecard development (when implemented)
cr-score run --config config.yml

# List runs
cr-score list-runs --limit 10

# Compare runs for reproducibility
cr-score compare --run-id-a run_123 --run-id-b run_456
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

### Completed (v0.1.0-alpha)

- Core infrastructure (config, registry, logging, hashing)
- Data layer (connectors, validation, quality checks)
- Spark layer (session factory, compression with sample weighting)
- CLI interface with validation, run management, comparison
- Comprehensive configuration schema with Pydantic validation

### In Progress

- EDA module
- Binning engine
- WoE encoding and reject inference
- Modeling, calibration, and scaling
- Reporting and visualization
- MCP tools
- Test suite
- Documentation

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
