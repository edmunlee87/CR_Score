================================================================================
CR_Score AGENT IMPLEMENTATION RULES
================================================================================

Version: 1.0
Date: 2026-01-15
Purpose: Define core principles and constraints for any agent implementing 
         the CR_Score scorecard development platform

================================================================================
1. DESIGN PRINCIPLE ENFORCEMENT
================================================================================

All implementations MUST embody the 5 core design principles:

P1: CONFIG-FIRST
   - Every action expressible as YAML/JSON configuration
   - No hardcoded values or magic numbers
   - All defaults in config schema with override capability
   - Guideline: If you write hardcoded logic, add a config parameter

P2: ARTIFACT-FIRST
   - Every execution step produces versioned, hashed artifacts
   - Artifacts indexed in artifact_index.json
   - All outputs serializable to JSON/CSV/Parquet
   - Guideline: No step complete until all outputs written to artifact store

P3: DETERMINISTIC BY DEFAULT
   - Same config + same data = always same result
   - All randomness seeded and logged
   - No temporal dependencies or external service calls
   - Guideline: Every module must be independently testable with fixed seed

P4: SPARK WHERE IT MATTERS
   - Use Spark for: heavy scans, aggregations, compression, stability analysis
   - Use Python for: orchestration, diagnostics, visualization, UX
   - Never duplicate cross-module logic across Spark and Python
   - Guideline: Implement 80/20 Spark first, Python fallback paths second

P5: SCALE WITHOUT LOSING CORRECTNESS
   - Row-level data optional once sufficient statistics available
   - Weighted aggregation preserves likelihoods and event rates
   - Compression must be mathematically verified
   - Guideline: All multi-row operations must support sample_weight parameter

================================================================================
2. MODULE ORGANIZATION & NAMING CONVENTIONS
================================================================================

DIRECTORY STRUCTURE:
   CR_Score/
   ├── core/           # Foundational infrastructure (config, registry, hashing, logging)
   ├── data/           # Data connectors, validation, optimization
   ├── spark/          # Spark-specific operations and optimizations
   ├── eda/            # Exploratory data analysis
   ├── features/       # Feature engineering and recipes
   ├── binning/        # Bin creation and management
   ├── encoding/       # WoE and scoring transformations
   ├── reject_inference/ # Missing data handling
   ├── model/          # Modeling and diagnostics
   ├── calibration/    # Score calibration
   ├── scaling/        # Score scaling and points table
   ├── explainability/ # Reason codes and adverse action
   ├── monitoring/     # Vintage tracking and monitoring
   ├── viz/            # Visualization and plots
   ├── reporting/      # Report generation
   ├── tools/          # MCP and tool interface
   ├── api/            # REST API
   ├── ui/             # Web UI components
   ├── cli/            # Command-line interface
   ├── templates/      # Project templates
   ├── tests/          # Test suite (unit, integration, spark, reproducibility)
   └── docs/           # Documentation and guides

FILE NAMING:
   - Python modules: lowercase_with_underscores.py
   - Classes: PascalCase
   - Functions: snake_case
   - Constants: UPPERCASE_WITH_UNDERSCORES
   - Test files: test_<module_name>.py
   - Fixtures: conftest.py per test directory level

IMPORT ORGANIZATION:
   1. Standard library imports
   2. Third-party imports (pandas, pyspark, etc.)
   3. Local imports from CR_Score package
   4. Separate groups with blank lines

================================================================================
3. CODE QUALITY GATES
================================================================================

MANDATORY REQUIREMENTS:

Typing:
   - All function signatures must include type hints (PEP 484)
   - Use typing.Optional, Union, List, Dict as needed
   - Use Protocol for structural typing where appropriate
   - Guideline: mypy --strict should pass for all modules

Docstrings:
   - All public functions/classes require docstrings
   - Format: Google-style docstrings with Args, Returns, Raises sections
   - Include example usage for complex functions
   - Document config parameters with defaults

Error Handling:
   - All I/O operations wrapped in try-except with specific error types
   - All external service calls include retry logic with backoff
   - Raise custom CR_ScoreException or subclasses
   - Never silently catch broad exceptions
   - Guideline: Every except clause must have a handler

Logging:
   - Import logger as: from CR_Score.core.logging import get_audit_logger
   - Use logger.info, logger.warning, logger.error appropriately
   - Log all config changes, overrides, and decisions at INFO level
   - Log errors with traceback at ERROR level
   - Use structured logging (JSON) for audit trail

Performance:
   - Memory profiling for large data operations
   - No nested loops over millions of items without vectorization
   - DataFrame operations must use vectorized methods, not .apply()
   - Profile with cProfile before deployment

================================================================================
4. TESTING REQUIREMENTS
================================================================================

UNIT TEST COVERAGE:

Minimum Coverage: 80% per module
   - All public functions tested
   - Happy path + error paths
   - Edge cases (empty inputs, None values, large values)
   - Parametrized tests for multiple scenarios

Test Organization:
   - tests/unit/<module_name>/test_<filename>.py
   - Separate test file per module file
   - Use pytest fixtures from conftest.py
   - Use pytest.mark.parametrize for data-driven tests

Fixture Standards:
   - conftest.py at each test directory level
   - Fixtures include: sample data, config templates, mock objects
   - Use @pytest.fixture(scope="session") for expensive setup
   - Guideline: All fixtures must be deterministic and reproducible

INTEGRATION TESTS:

Cross-Module Data Flow:
   - tests/integration/test_<workflow>.py
   - Test full pipeline sections (e.g., config → binning → encoding)
   - Use golden reference data
   - Verify artifact generation and indexing

SPARK TESTS:

Spark Equivalence:
   - tests/spark/test_<spark_module>.py
   - Compare Spark output to pandas equivalent for small data
   - Verify skew handling and partition correctness
   - Test checkpoint and persistence strategies

REPRODUCIBILITY TESTS:

Golden Artifacts:
   - tests/reproducibility/test_golden_runs.py
   - Store canonical artifacts in tests/golden_artifacts/
   - Compare new runs against golden via artifact hashing
   - Tolerance defined in test config (default: 0.0 for logits, 0.001 for scores)

================================================================================
5. ARTIFACT GENERATION CHECKLIST
================================================================================

EVERY STEP MUST:

☐ Produce versioned output files (JSON/CSV/Parquet)
☐ Generate content hash (SHA256) of all outputs
☐ Create entry in artifact_index.json with:
   - artifact_id (unique, deterministic from step ID)
   - artifact_type (e.g., "binning_table", "run_metadata")
   - content_hash
   - file_paths
   - timestamp
   - schema or structure definition
   - lineage (input artifact IDs)

☐ Validate artifact integrity:
   - File exists and is readable
   - Content matches declared schema
   - Hash matches expected value

☐ Log artifact creation:
   - artifact_created event with all metadata
   - Include run_id, step_id, user_id
   - Record any manual overrides or configuration changes

☐ Update run_metadata.json:
   - Add step execution time
   - Add memory/CPU usage
   - Add data volumes (rows in, rows out, compression ratio)

Artifact Naming Convention:
   - {run_id}/{step_id}_{artifact_type}.{ext}
   - Example: run_20260115_001/03_binning_tables.parquet
   - Archive: artifacts/{project_id}/{run_id}/

================================================================================
6. CONFIGURATION MANAGEMENT
================================================================================

CONFIG FILES:

Primary Config:
   - Format: YAML (config.yml)
   - Validated against schema.py
   - All possible parameters documented with defaults
   - Version field required (matches URD version)

Defaults:
   - CR_Score/core/config/defaults.yml
   - Overridable via environment variables: CR_Score_*
   - Overridable via CLI flags or SDK parameters

Schema Validation:
   - All inputs validated on load via Pydantic
   - Coercion rules documented
   - Validation errors raised with helpful messages
   - Guideline: No config should silently default if invalid

Config Sections:
   project, execution, data, target, split, data_optimization,
   compression, sampling, eda, features, binning, reject_inference,
   model, calibration, scaling, reporting, interfaces, tools

================================================================================
7. PERMISSION & AUDIT REQUIREMENTS
================================================================================

ROLE-BASED ACCESS:

Roles:
   - Viewer: Read-only access to runs and reports
   - Analyst: Create configs, run EDA, view binning
   - Modeler: Full modeling capabilities, manual overrides
   - Validator: Reproducibility testing, run locking
   - Admin: User management, system configuration

Permission Checks:
   - All functions with @require_permission decorator
   - Permissions enforced at API/CLI/SDK level
   - Decision logged with user_id and timestamp

Manual Overrides:
   - Require Modeler+ role
   - Log reason + full diff of change
   - Validator receives notification
   - Cannot be undone (immutable history)

Audit Logging:
   - All decisions logged to audit_log.jsonl
   - Format: {timestamp, user_id, action, resource_id, before, after, reason}
   - Queryable by run_id, user_id, action type, date range
   - Retained indefinitely (compliance requirement)

================================================================================
8. PERFORMANCE & SCALABILITY RULES
================================================================================

Memory Management:
   - Spark DataFrame operations should not materialize full datasets
   - Use lazy evaluation and checkpointing
   - Cache only intermediate results needed multiple times
   - Profile memory usage for operations >100M rows

Parallelization:
   - Spark shuffle_partitions set via config (default: 800)
   - Use partition-aware aggregations
   - Detect and handle skew with salting
   - Target: Linear scaling to 100M+ rows

Caching:
   - Cache by (data_hash, config_hash) tuple
   - Store in configurable location (local, S3)
   - Invalidate when input data or config changes
   - Guideline: Caching should reduce runtime by >50% for common operations

================================================================================
9. DOCUMENTATION STANDARDS
================================================================================

Code Documentation:
   - Docstrings for all public APIs
   - README.md in each major module directory
   - Example usage in docstrings and docs/examples/
   - Architecture decisions in docs/adr/ (Architecture Decision Records)

User Documentation:
   - User manual in docs/user_manual/
   - API reference auto-generated from docstrings
   - Validation guides for each scorecard type
   - Troubleshooting guide with common errors

Model Documentation:
   - Auto-generated model cards (model_card.md)
   - Include: model type, target definition, features, performance, limitations
   - Include: training data characteristics, deployment constraints
   - Include: monitoring plan and performance targets

================================================================================
10. VERSIONING & RELEASE STRATEGY
================================================================================

Semantic Versioning:
   - Format: MAJOR.MINOR.PATCH (e.g., 1.0.0)
   - MAJOR: Breaking changes to config schema or artifact format
   - MINOR: New features, backward compatible
   - PATCH: Bug fixes

Artifact Versioning:
   - Include CR_Score_version in all artifacts
   - Include schema_version for each artifact type
   - Document breaking changes in CHANGELOG.md

Backward Compatibility:
   - Support reading artifacts from previous 2 MINOR versions
   - Provide migration scripts for config schema changes
   - Document deprecation path for removed features (1 MINOR version notice)

================================================================================
11. EMERGENCY RULES (BREAK GLASS IF NEEDED)
================================================================================

If you encounter a situation where these rules prevent implementation:

1. Document the constraint violation: REASON, IMPACT, PROPOSED EXCEPTION
2. Minimize scope: Affect smallest possible codebase section
3. Add TODO with reference to URD section requiring this
4. Log decision in ADR (Architecture Decision Record)
5. Plan immediate refactor to re-comply

Examples:
   - Hardcoded value for performance: Add config parameter in next sprint
   - Missing test for infrastructure code: Add test as bug fix in next week
   - Determinism violation for legitimiate reason: Document in code + ADR

================================================================================
