================================================================================
CR_Score VALIDATION GATES
================================================================================

Version: 1.0
Date: 2026-01-15
Purpose: Define quality checkpoints and acceptance criteria for code integration

================================================================================
1. PRE-COMMIT VALIDATION (Developer Local)
================================================================================

Run BEFORE committing code:

    # Command:
    make pre-commit
    
    # Includes:
    ☐ Linting (flake8, pylint)
    ☐ Type checking (mypy --strict)
    ☐ Import sorting (isort)
    ☐ Code formatting (black --line-length 100)
    ☐ Security scanning (bandit)
    ☐ Docstring validation (pydocstyle)

Linting Rules:

    Tool: flake8 + pylint
    Config: .flake8, .pylintrc in repo root
    
    Ignored codes:
    - W503: Line break before binary operator (allowed in CR_Score)
    
    Max complexity: 10 (McCabe cyclomatic complexity)
    Max line length: 100
    
    Errors (fail immediately):
    - E741: ambiguous variable names
    - E901: SyntaxError
    - E902: IOError
    - F403: 'from module import *' (wildcard imports)

Type Checking:

    Tool: mypy --strict
    Config: mypy.ini
    
    All functions must have return type hints
    All function parameters must have type hints
    Use Optional[T] for nullable values
    Use Protocol for structural typing
    
    Allowed mypy overrides (with # type: ignore comment):
    - Third-party libraries with no stubs
    - Spark API quirks
    
    Must include explanation in comment:
    # type: ignore  # Spark DataFrame has no type stubs

Import Organization:

    Tool: isort --profile black
    Config: .isort.cfg
    
    Groups (in order):
    1. __future__ imports
    2. Standard library (sys, os, json, etc.)
    3. Third-party (pandas, pyspark, fastapi, etc.)
    4. Local CR_Score imports
    5. Relative imports (if allowed)
    
    Within group: Alphabetical order
    Blank line between groups
    
    Bad: from CR_Score import * (use explicit imports)

Code Formatting:

    Tool: black --line-length 100
    Config: pyproject.toml
    
    Must pass auto-formatting:
    black CR_Score/
    
    Don't override black formatting (only in documented exceptions)

Docstring Validation:

    Tool: pydocstyle
    Config: .pydocstyle
    
    All public functions/classes require docstrings
    No module docstrings starting with triple quotes only (need description)
    Args/Returns sections required for public APIs

================================================================================
2. UNIT TEST COVERAGE GATES
================================================================================

Minimum Coverage Requirement: 80% per module

Run Tests:

    # Command:
    pytest tests/unit/ --cov=CR_Score --cov-report=html
    
    # Exit on failure:
    pytest tests/unit/ --cov=CR_Score --cov-fail-under=80

Coverage Rules:

    Measured: Line coverage (not branch coverage)
    Exclude: Abstract base classes, __main__ blocks, __init__.py boilerplate
    Minimum: 80% for production code
    Target: 90% for critical modules (compression, encoding, model)
    
    Per-module minimum:
    ├── core/           (95% - foundational)
    ├── data/           (85% - validation critical)
    ├── spark/          (85% - concurrency critical)
    ├── binning/        (90% - logic-heavy)
    ├── encoding/       (95% - correctness critical)
    ├── model/          (85% - results-driven)
    └── [other]/        (80% - baseline)

Test Scenarios Required:

    Happy Path:
    ☐ Normal input → expected output
    ☐ Default parameters work correctly
    ☐ Config validation passes
    
    Error Paths:
    ☐ None/null inputs → appropriate error
    ☐ Empty collections → handled gracefully
    ☐ Invalid config → ConfigurationError
    ☐ Missing files → FileNotFoundError
    ☐ Permission denied → PermissionError
    
    Edge Cases:
    ☐ Boundary values (min, max)
    ☐ Single-element collections
    ☐ Large values (>1B numbers)
    ☐ Unicode/special characters in strings
    ☐ Floating-point precision edge cases
    
    Determinism:
    ☐ Same input + same seed → identical output
    ☐ Run twice → exact same results
    ☐ No temporal dependencies

Test Quality Checklist:

    ☐ Test names describe what they test (test_<function>_<scenario>)
    ☐ Each test focuses on one thing (single assertion preferred)
    ☐ Fixtures are reusable and deterministic
    ☐ No test interdependencies (can run in any order)
    ☐ Setup/teardown properly isolates tests
    ☐ Mocks only mock external dependencies
    ☐ Parameterized tests for data-driven scenarios
    ☐ Comments explain non-obvious test logic

================================================================================
3. INTEGRATION TEST GATES
================================================================================

Cross-Module Data Flow Validation:

    Test Location: tests/integration/
    Run Frequency: Before merge to main
    Command: pytest tests/integration/ -v
    
    Required Coverage:
    ☐ Config → Data Loading → Validation
    ☐ Validation → EDA → Feature Selection
    ☐ Features → Binning → Encoding
    ☐ Encoding → Model Training → Calibration
    ☐ Calibration → Scaling → Export

Artifact Chain Validation:

    ☐ Each step produces expected artifact types
    ☐ Artifacts match declared schema
    ☐ Artifact hashes are reproducible
    ☐ Artifact lineage correctly captured
    ☐ Downstream steps can read upstream artifacts
    ☐ No data loss between steps

Integration Test Template:

    def test_full_scorecard_pipeline(config, sample_data, run_id):
        """Test end-to-end scorecard from data to export."""
        
        # Step 1: Validation
        validate_data(sample_data, config.data_contract)
        
        # Step 2: EDA
        eda_results = run_eda(sample_data, config.eda, run_id)
        assert eda_results["status"] == "success"
        
        # Step 3: Binning
        bins = process_binning(sample_data, config.binning, run_id)
        assert len(bins) > 0
        
        # Step 4: Encoding
        encoded_df = apply_woe_encoding(sample_data, bins, run_id)
        assert encoded_df.count() == sample_data.count()
        
        # Step 5: Model
        model = train_scorecard(encoded_df, config.model, run_id)
        assert model.r2_score() > 0.5
        
        # Step 6: Scaling
        scaled_scores = scale_scores(model, config.scaling, run_id)
        assert scaled_scores["base_score"] > 0
        
        # Verify lineage
        final_artifacts = artifact_index.list_by_run(run_id)
        assert all(
            final_artifacts[i]["lineage"] 
            for i in final_artifacts
        )

Performance Gates:

    ☐ Integration tests complete in < 5 minutes
    ☐ No memory leaks detected by tracemalloc
    ☐ Spark session properly cleanup (no orphaned processes)
    ☐ Temp files cleaned up after tests

================================================================================
4. SPARK EQUIVALENCE TESTS
================================================================================

Spark vs. Python Correctness:

    Location: tests/spark/
    Run: pytest tests/spark/ --spark-local
    
    Pattern:
    1. Execute operation on small pandas DataFrame
    2. Execute same operation on Spark DataFrame
    3. Compare results (must be mathematically identical)
    
    Example:
    
        def test_compression_equivalence(small_data, config):
            """Verify Spark compression matches pandas."""
            
            # Python path
            pdf = small_data  # pandas.DataFrame
            compressed_py = compress_pandas(pdf, config)
            
            # Spark path
            sdf = spark.createDataFrame(small_data)
            compressed_spark = compress_spark(sdf, config).toPandas()
            
            # Compare
            assert_frame_equal(
                compressed_py.sort_values("bin_id"),
                compressed_spark.sort_values("bin_id"),
                check_dtype=False,
                atol=1e-10
            )

Partition Correctness:

    ☐ Results identical regardless of partition count
    ☐ Skew detection identifies problematic keys
    ☐ Salting strategy redistributes data evenly
    ☐ Checkpoint/cache doesn't change results

Numerical Precision:

    ☐ Event rates match to 1e-10 relative tolerance
    ☐ Logistic regression coefficients match to 1e-8
    ☐ Aggregate sums exact (no floating-point drift)
    ☐ Weighted aggregations preserve likelihood

================================================================================
5. REPRODUCIBILITY TEST GATES
================================================================================

Golden Artifacts Repository:

    Location: tests/golden_artifacts/
    Structure:
    tests/golden_artifacts/
    ├── v1.0/
    │   ├── config.yml
    │   ├── data_snapshot_ref.json
    │   ├── run_metadata.json
    │   ├── eda_summary.json
    │   ├── binning_tables/
    │   ├── model_coefficients.csv
    │   ├── scaling/
    │   └── final_report.html
    └── v1.2/
        └── ... (updated golden for v1.2)

Reproducibility Verification:

    Command: pytest tests/reproducibility/test_golden_runs.py
    Frequency: Every commit to main
    
    Steps:
    1. Load golden config from tests/golden_artifacts/
    2. Execute full pipeline with same config
    3. Compare outputs against golden artifacts
    4. Compute differences:
       - Exact match: Content hash identical
       - Tolerance: Event rates, scores within tol
       - Warnings: Acceptable differences (dependency versions)

Tolerance Thresholds:

    Event Rates: ±0.0001 (0.01% relative)
    Score Values: ±1 point (on 0–999 scale)
    Coefficients: ±1e-8 (relative)
    IV Values: ±1e-6 (relative)
    
    Reason for tolerance:
    - Spark executor variance in floating-point arithmetic
    - Library version updates (numpy, scipy)
    - Multi-threaded aggregation rounding
    
    Zero Tolerance (must be exact):
    - Artifact counts (rows in, rows out)
    - Event sums (aggregates)
    - Missing value patterns

Failure Response:

    If reproducibility test fails:
    1. Identify delta between runs
    2. Investigate root cause:
       - Config changed? (OK, document)
       - Data changed? (Expected, update golden)
       - Code changed? (Debug, ensure correctness)
       - Dependency changed? (May be OK, check impact)
    3. If acceptable change:
       - Update golden artifacts
       - Document reason in CHANGELOG.md
       - Re-run tests to verify
    4. If unexpected change:
       - Revert code changes
       - Fix and debug
       - Re-run full validation

Artifact Comparison Report:

    Output (auto-generated):
    tests/reproducibility/comparison_report.html
    
    Includes:
    - Diff of metadata (timestamps OK, content hash critical)
    - Diff of numeric outputs (with tolerance visualization)
    - Files added/removed
    - Lineage verification
    - Performance metrics (execution time, memory usage)

================================================================================
6. CONFIGURATION VALIDATION GATES
================================================================================

Schema Validation:

    ☐ All config fields match schema definition
    ☐ Type coercion works (string "true" → bool)
    ☐ Required fields present (no missing)
    ☐ Optional fields have defaults
    ☐ Bounds checked (max_bins >= 2, shuffle_partitions > 0)
    ☐ Enum fields in allowed values
    ☐ Relationships validated (dev_period < oot_period)

Config Completeness:

    ☐ Can load defaults from CR_Score/core/config/defaults.yml
    ☐ Can override via environment variables (CR_Score_*)
    ☐ Can override via CLI flags
    ☐ Can override via SDK parameters
    ☐ Precedence: CLI flags > env vars > provided config > defaults

Configuration Error Handling:

    ☐ Invalid config raises ConfigurationError (not ValueError)
    ☐ Error message identifies problematic field
    ☐ Error message suggests valid values/ranges
    ☐ Error includes line number in YAML if applicable
    
    Example error:
    ConfigurationError: Invalid config at compression.mode: 
    "invalid_mode" not in ["post_binning_exact", "eda_sufficient_stats", 
    "hybrid_topk_tail"]. Did you mean "post_binning_exact"?

Cross-Field Validation:

    ☐ If compress.enabled=true, compress.mode must be set
    ☐ If reject_inference.method="parceling", parceling params required
    ☐ If execution.engine="spark_cluster", spark config must be present
    ☐ If binning.enforce_monotonic=true, method must be "monotonic" compatible

================================================================================
7. PERMISSION & AUDIT VALIDATION
================================================================================

Access Control Tests:

    ☐ Viewer role cannot modify binning
    ☐ Analyst role cannot access Admin functions
    ☐ Manual override requires Modeler+ role
    ☐ Validator cannot bypass immutability
    ☐ All permission denials logged to audit trail

Audit Trail Validation:

    ☐ All decisions recorded in audit_log.jsonl
    ☐ Each entry has: timestamp, user_id, action, resource_id, reason
    ☐ Before/after values captured for mutations
    ☐ Audit log immutable (append-only)
    ☐ Audit log queryable by run_id, user_id, date range
    ☐ Audit log entries have correlation IDs

Manual Override Tracking:

    ☐ Override captured with: user_id, timestamp, old_value, new_value, reason
    ☐ Reason required and logged (cannot be empty)
    ☐ Validator notified of overrides (notification logged)
    ☐ Override diff logged in run_metadata

================================================================================
8. PERFORMANCE GATES
================================================================================

Execution Time Targets:

    Operation                | Data Size    | Target Time
    -------------------------|--------------|------------
    Config validation         | Any          | < 100ms
    Column pruning            | 100M rows    | < 2 sec
    Type optimization         | 100M rows    | < 3 sec
    Binning (single variable) | 100M rows    | < 30 sec
    Compression               | 100M rows    | < 20 sec
    WoE encoding              | 100M rows    | < 10 sec
    Logit model training      | 100M rows    | < 5 min
    Full pipeline             | 100M rows    | < 20 min

Memory Usage Targets:

    ☐ Python process: < 4 GB for orchestration
    ☐ Spark executor: Configurable, default 2 GB
    ☐ Temp files: Cleaned up within 24 hours
    ☐ Cache eviction: Auto-clear when > 80% capacity

Compression Effectiveness:

    ☐ Compression ratio: Target >10x for typical data
    ☐ Data preservation: 100% event preservation after compression
    ☐ Verification overhead: < 5% of compression time

Scalability Tests:

    Run with datasets: 1M, 10M, 100M, 1B rows
    ☐ Linear scaling to 100M rows
    ☐ Execution time grows < 2x for 10x rows
    ☐ Memory usage remains stable (not proportional to rows)

================================================================================
9. SECURITY GATES
================================================================================

Dependency Scanning:

    Tool: safety check
    Command: safety check --json
    ☐ No known vulnerabilities in dependencies
    ☐ Dependency versions locked in requirements.txt
    ☐ Regular updates checked (weekly CI job)

Code Security Scanning:

    Tool: bandit
    Command: bandit -r CR_Score/
    ☐ No hardcoded secrets
    ☐ No SQL injection vectors
    ☐ No insecure deserialization
    ☐ No unsafe temporary files

Data Privacy:

    ☐ No PII logged (SSN, credit card, phone)
    ☐ Data masking applied to audit logs
    ☐ Sensitive columns marked in schema contract
    ☐ Export masks sensitive data by default

================================================================================
10. MERGE GATE CHECKLIST
================================================================================

Before merging PR to main branch:

    Code Quality:
    ☐ Pre-commit checks pass (linting, formatting, types)
    ☐ Unit test coverage ≥ 80% new code
    ☐ All unit tests pass
    ☐ No new violations of AGENT_RULES.md
    
    Integration:
    ☐ Integration tests pass
    ☐ Artifact schemas validated
    ☐ Lineage verified
    ☐ No breaking changes to config schema
    
    Performance:
    ☐ Spark equivalence tests pass
    ☐ Reproducibility tests pass (within tolerance)
    ☐ No performance regression > 10%
    ☐ Memory usage stable
    
    Compliance:
    ☐ Docstrings added/updated
    ☐ Audit logging implemented
    ☐ Permissions checked
    ☐ CHANGELOG.md updated
    
    CI/CD:
    ☐ GitHub Actions pipeline succeeds
    ☐ All check status green
    ☐ Code review approved (2+ reviewers for core/)
    ☐ No merge conflicts
    
    PR Quality:
    ☐ Description explains changes
    ☐ Commits are atomic and well-message
    ☐ No debugging code or comments
    ☐ References related issues

Automated vs. Manual Checks:

    Automated (GitHub Actions):
    - Pre-commit checks
    - Unit tests + coverage
    - Integration tests
    - Spark tests
    - Reproducibility tests (golden artifacts)
    - Security scanning
    - Build Docker image
    
    Manual (Code Review):
    - Architectural correctness
    - Design pattern compliance
    - Documentation quality
    - Potential performance issues
    - Edge case handling

================================================================================
