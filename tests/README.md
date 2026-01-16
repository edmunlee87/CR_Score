# CR_Score Test Suite

Comprehensive test coverage for all CR_Score modules.

## Test Structure

```
tests/
├── unit/                # Unit tests for individual modules
│   ├── test_models.py          # All 4 model families
│   ├── test_evaluation.py      # Evaluation metrics
│   ├── test_reporting.py       # Report export
│   ├── test_monitoring.py      # Monitoring modules
│   ├── test_feature_selection.py
│   ├── test_pipeline.py
│   └── test_woe_encoding.py
├── integration/         # Integration tests
├── reproducibility/     # Reproducibility tests
├── spark/              # Spark-specific tests
└── conftest.py         # Shared fixtures
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Suite
```bash
pytest tests/unit/
pytest tests/unit/test_models.py
pytest tests/unit/test_evaluation.py
```

### Run with Coverage
```bash
pytest tests/ --cov=src/cr_score --cov-report=html --cov-report=term
```

### Run Specific Test Class
```bash
pytest tests/unit/test_models.py::TestLogisticScorecard
```

### Run Specific Test
```bash
pytest tests/unit/test_models.py::TestLogisticScorecard::test_fit
```

## Test Coverage

### New Modules (60+ tests)

#### Model Families (test_models.py)
- ✅ LogisticScorecard (12 tests)
  - Initialization, fit, predict, predict_proba
  - Sample weights, coefficients, feature importance
  - Performance metrics, export
- ✅ RandomForestScorecard (5 tests)
  - Initialization, fit, predict
  - Feature importance, tree depth statistics
- ✅ XGBoostScorecard (2 tests, optional dependency)
  - Initialization, fit and predict
- ✅ LightGBMScorecard (2 tests, optional dependency)
  - Initialization, fit and predict
- ✅ Sklearn Compatibility (2 tests)
  - clone() support
  - cross_val_score() support

#### Evaluation Module (test_evaluation.py)
- ✅ ClassificationMetrics (6 tests)
  - Accuracy, precision, recall, F1, MCC
  - Confusion matrix, optimal threshold
- ✅ StabilityMetrics (5 tests)
  - PSI calculation and interpretation
  - CSI calculation
  - Feature-level stability
- ✅ CalibrationMetrics (4 tests)
  - Brier score, log loss, ECE
  - Calibration curve
- ✅ RankingMetrics (6 tests)
  - AUC, Gini, KS statistic
  - Lift curve, gains curve
- ✅ PerformanceEvaluator (4 tests)
  - Comprehensive evaluation
  - Stability evaluation
  - Summary generation
  - Model comparison

#### Reporting Module (test_reporting.py)
- ✅ ReportExporter (6 tests)
  - JSON export
  - CSV export (multiple files)
  - Excel export (multi-sheet)
  - Markdown export
  - Comprehensive report generation
  - Serialization utilities

#### Monitoring Module (test_monitoring.py)
- ✅ PerformanceMonitor (4 tests)
  - Initialization with baselines
  - Recording predictions
  - Health check
  - Metrics summary
- ✅ DriftMonitor (3 tests)
  - Initialization
  - Drift detection
  - Drift summary
- ✅ AlertManager (4 tests)
  - Alert creation
  - Getting active alerts
  - Resolving alerts
  - Alert summary
- ✅ MetricsCollector (6 tests)
  - Counter increment
  - Gauge setting
  - Histogram recording
  - Getting metrics
  - Resetting metrics

### Coverage Goals

- **Target:** 70%+ overall coverage
- **Critical Modules:** 80%+ coverage
  - Model families: 85%
  - Evaluation: 90%
  - Reporting: 75%
  - Monitoring: 80%

## CI/CD Integration

### GitHub Actions Workflow

The test suite runs automatically on:
- Push to main branch
- Pull requests
- Scheduled daily runs

Workflow includes:
1. **Multi-Python Testing** (3.9, 3.10, 3.11)
2. **Linting** (flake8, black)
3. **Coverage Report** (pytest-cov)
4. **Optional Dependencies** (graceful skipping)

### Badges

Add to README.md:
```markdown
![Tests](https://github.com/edmunlee87/CR_Score/workflows/CI/badge.svg)
![Coverage](https://codecov.io/gh/edmunlee87/CR_Score/branch/main/graph/badge.svg)
```

## Test Fixtures

### Common Fixtures (conftest.py)

- `sample_data`: Binary classification data (1000 samples, 10 features)
- `sample_metrics`: Mock performance metrics
- `sample_model`: Mock model for testing
- `baseline_metrics`: Baseline metrics for monitoring

## Optional Dependencies

Tests handle optional dependencies gracefully:

```python
@pytest.mark.skipif(
    not pytest.importorskip("xgboost", minversion=None),
    reason="XGBoost not installed"
)
def test_xgboost_feature():
    # Test only runs if XGBoost is installed
    pass
```

## Writing New Tests

### Test Naming Convention
- File: `test_<module_name>.py`
- Class: `Test<ClassName>`
- Method: `test_<what_it_tests>`

### Example Test
```python
class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Arrange
        feature = NewFeature(param=value)
        
        # Act
        result = feature.process(sample_data)
        
        # Assert
        assert result is not None
        assert len(result) > 0
```

### Using Fixtures
```python
@pytest.fixture
def custom_data():
    """Generate custom test data."""
    return pd.DataFrame({'col1': [1, 2, 3]})

def test_with_fixture(custom_data):
    """Test using custom fixture."""
    assert len(custom_data) == 3
```

## Continuous Integration

### Local Pre-commit Checks
```bash
# Run tests
pytest tests/

# Check coverage
pytest tests/ --cov=src/cr_score

# Run linting
flake8 src/
black src/ --check
```

### CI/CD Requirements
- All tests must pass
- Coverage must be >= 70%
- No linting errors
- Documentation builds successfully

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed
   pip install -e .
   ```

2. **Optional Dependency Tests Failing**
   ```bash
   # Install optional dependencies
   pip install xgboost lightgbm
   ```

3. **Coverage Not Working**
   ```bash
   # Install pytest-cov
   pip install pytest-cov
   ```

4. **Tests Run Slowly**
   ```bash
   # Run in parallel
   pytest tests/ -n auto
   ```

## Test Metrics

### Current Status
- **Total Tests:** 60+ test methods
- **Test Files:** 7 files
- **Coverage:** Target 70%+ (actual TBD after CI run)
- **Duration:** <30 seconds (unit tests only)

### Module Coverage
| Module | Tests | Status |
|--------|-------|--------|
| Models | 23 | ✅ Complete |
| Evaluation | 25 | ✅ Complete |
| Reporting | 6 | ✅ Complete |
| Monitoring | 17 | ✅ Complete |
| Feature Selection | 3 | ⚠️  Needs update |
| Pipeline | 2 | ⚠️  Needs update |
| WoE Encoding | 9 | ⚠️  Needs API fix |

## Next Steps

1. ✅ Add tests for new model families
2. ✅ Add tests for evaluation module
3. ✅ Add tests for reporting module
4. ✅ Add tests for monitoring module
5. 🔄 Update existing tests for current API
6. 🔄 Add integration tests
7. 🔄 Add reproducibility tests
8. 🔄 Achieve 70%+ coverage

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass locally
3. Check coverage locally
4. Submit PR with tests included
5. Wait for CI/CD to pass

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
