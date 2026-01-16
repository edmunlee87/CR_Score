# CR_Score Configuration Templates

Ready-to-use configuration templates for different use cases and skill levels.

## Template Levels

### Beginner Templates
**For:** First-time users, simple scorecards  
**Features:** Minimal configuration, optimal defaults  
**Use when:** Learning CR_Score, prototyping quickly

**Files:**
- `basic_scorecard.yml` - Simple credit scorecard with essential settings only

### Intermediate Templates
**For:** Production-ready scorecards  
**Features:** Full EDA, feature selection, comprehensive reporting  
**Use when:** Building real scorecards for production

**Files:**
- `full_scorecard.yml` - Complete scorecard with all standard features

### Advanced Templates
**For:** Enterprise deployments  
**Features:** Monitoring, observability, compliance, governance  
**Use when:** Deploying at scale with full MLOps

**Files:**
- `production_scorecard.yml` - Enterprise-grade with full monitoring

## Usage

### From Code

```python
from cr_score.templates import get_template_path
import yaml

# Load template
template_path = get_template_path('beginner', 'basic_scorecard.yml')
with open(template_path) as f:
    config = yaml.safe_load(f)

# Customize config
config['project']['name'] = 'my_scorecard'
config['data']['train_path'] = 'my_data.csv'

# Use with CR_Score
from cr_score import ScorecardPipeline
pipeline = ScorecardPipeline.from_config(config)
```

### From CLI

```bash
# List available templates
cr-score list-templates

# Initialize from template
cr-score init --template beginner/basic_scorecard.yml --output my_config.yml

# Run with template
cr-score run --config beginner/basic_scorecard.yml
```

## Customization Guide

### Beginner Level
Modify only these sections:
- `project.name` - Your project name
- `data` paths - Your data files
- `scaling` parameters - Your score range

### Intermediate Level
Additionally customize:
- `features.selection` - Feature selection method
- `binning` - Binning strategy
- `model` - Model parameters

### Advanced Level
Full customization including:
- `monitoring` - Set up monitoring
- `explainability` - Enable SHAP, reason codes
- `governance` - Compliance settings
- `interfaces` - API, UI configuration

## Feature Comparison

| Feature | Beginner | Intermediate | Advanced |
|---------|----------|--------------|----------|
| Basic Scorecard | ✅ | ✅ | ✅ |
| EDA | ❌ | ✅ | ✅ |
| Feature Selection | ❌ | ✅ | ✅ |
| Optimal Binning | ✅ | ✅ | ✅ |
| Calibration | ❌ | ✅ | ✅ |
| Monitoring | ❌ | ❌ | ✅ |
| Drift Detection | ❌ | ❌ | ✅ |
| Explainability | ❌ | ❌ | ✅ |
| Alert Management | ❌ | ❌ | ✅ |
| Metrics Collection | ❌ | ❌ | ✅ |
| Observability Dashboard | ❌ | ❌ | ✅ |

## Configuration Sections

### Core Sections (All Levels)
- `project` - Project metadata
- `data` - Data sources
- `binning` - Binning configuration
- `model` - Model settings
- `scaling` - PDO scaling

### Intermediate Sections
- `eda` - Exploratory data analysis
- `features` - Feature engineering
- `calibration` - Model calibration
- `viz` - Visualization settings

### Advanced Sections
- `monitoring.performance` - Performance monitoring
- `monitoring.drift` - Drift detection
- `monitoring.predictions` - Prediction monitoring
- `monitoring.alerts` - Alert configuration
- `monitoring.metrics` - Metrics collection
- `explainability` - Model explainability
- `governance` - Compliance and audit
- `interfaces` - API/UI configuration

## Best Practices

1. **Start Simple**: Begin with beginner template, gradually add features
2. **Version Control**: Keep config files in git
3. **Environment-Specific**: Use different configs for dev/staging/prod
4. **Document Changes**: Comment your customizations
5. **Test Thoroughly**: Validate config before production deployment

## Monitoring & Observability (Advanced)

The advanced template includes comprehensive monitoring:

### Performance Monitoring
Tracks model metrics over time, detects degradation:
- AUC, Precision, Recall, F1
- Baseline comparison
- Automated alerting

### Drift Detection
Monitors feature and prediction drift:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Per-feature drift tracking

### Prediction Monitoring
Tracks prediction distributions:
- Score range validation
- Anomaly detection
- Distribution shifts

### Alert Management
Configurable alerting system:
- Multiple severity levels
- Customizable thresholds
- Integration with notification systems

### Metrics Collection
Observability metrics in standard formats:
- Prometheus export
- Custom metrics
- Performance counters

### Observability Dashboard
Interactive HTML dashboards:
- Real-time metrics
- Historical trends
- Alert visualization

## Support

For questions or issues with templates:
1. Check the main CR_Score documentation
2. Review example notebooks in `playbooks/`
3. Open an issue on GitHub

## Contributing

To contribute new templates:
1. Follow the existing structure
2. Add comprehensive comments
3. Test thoroughly
4. Update this README
