================================================================================
CR_Score ARTIFACT SPECIFICATION
================================================================================

Version: 1.0
Date: 2026-01-15
Purpose: Define schema and format for all machine-readable artifacts

================================================================================
1. ARTIFACT VERSIONING & METADATA
================================================================================

Every artifact includes metadata:

    artifact_metadata.json (accompanies each artifact):
    {
        "artifact_id": "run_20260115_001/03_binning_tables",
        "artifact_type": "binning_table",
        "schema_version": "1.0",
        "CR_Score_version": "1.0.0",
        "file_paths": [
            "run_20260115_001/03_binning_tables_age.json",
            "run_20260115_001/03_binning_tables_income.json"
        ],
        "content_hash": {
            "algorithm": "sha256",
            "hex_digest": "a1b2c3d4e5f6..."
        },
        "timestamp_utc": "2026-01-15T14:30:45Z",
        "created_by": "CR_Score/v1.0.0",
        "run_id": "run_20260115_001",
        "step_id": "03_binning",
        "lineage": [
            "run_20260115_001/01_eda",
            "run_20260115_001/02_features"
        ],
        "data_lineage": {
            "input_rows": 5000000,
            "output_rows": 1250000,
            "compression_ratio": 4.0
        },
        "execution_metadata": {
            "execution_time_seconds": 47.3,
            "engine": "spark_cluster",
            "spark_app_id": "app-20260115-001234",
            "memory_peak_gb": 4.2
        },
        "validation_status": "passed",
        "validation_report": "Artifact hash verified, schema valid"
    }

Versioning Strategy:

    MAJOR version: Breaking schema changes (new required field, removed field)
    MINOR version: Backward-compatible additions (new optional field)
    PATCH version: Content updates (same schema, different values)
    
    Example progression:
    - v1.0: Initial schema
    - v1.1: Add optional field "confidence_interval"
    - v2.0: Change required field "event_weight" to "event_count" (breaking)

Backward Compatibility:

    When reading old artifact versions:
    1. Check schema_version
    2. Validate against compatibility matrix (stored in docs/)
    3. Apply migration if needed (e.g., rename fields)
    4. Log version mismatch at INFO level
    5. Proceed if compatible, raise SchemaError if incompatible
    
    Compatibility rule:
    - Can read schema versions within 1 MINOR version backward
    - Cannot read across MAJOR versions without migration

Content Hash Computation:

    For single file:
    1. Read entire file as bytes
    2. Compute SHA256(file_bytes)
    3. Store hex digest in content_hash.hex_digest
    
    For multi-file artifact:
    1. Sort file paths alphabetically
    2. Concatenate: SHA256(file_1) + SHA256(file_2) + ...
    3. Compute SHA256(concatenation)
    4. Store as combined hash
    5. Also store individual file hashes in "file_hashes" array

================================================================================
2. RUN METADATA ARTIFACT
================================================================================

run_metadata.json (central artifact for each run):

    {
        "run_id": "run_20260115_001",
        "project_id": "credit_scorecard_v2",
        "created_timestamp": "2026-01-15T10:00:00Z",
        "completed_timestamp": "2026-01-15T12:30:45Z",
        "status": "completed",  // "running", "completed", "failed", "cancelled"
        "initiated_by": "analyst_user_id",
        
        "config_snapshot": {
            // Full config used (see APPENDIX B of URD)
            "project": {...},
            "execution": {...},
            "data": {...},
            ...
        },
        "config_hash": "sha256_hex_digest_of_config_used",
        
        "data_snapshot_ref": {
            "source": "s3://bucket/data/20260101_snapshot.parquet",
            "data_hash": "sha256_of_data",
            "row_count": 5000000,
            "column_count": 150
        },
        
        "steps_executed": [
            {
                "step_id": "01_eda",
                "step_name": "Exploratory Data Analysis",
                "status": "completed",
                "start_time": "2026-01-15T10:01:00Z",
                "end_time": "2026-01-15T10:15:30Z",
                "execution_time_seconds": 925.3,
                "artifacts_created": [
                    "run_20260115_001/01_eda_summary.json",
                    "run_20260115_001/01_eda_report.html"
                ]
            },
            {
                "step_id": "02_features",
                "step_name": "Feature Engineering",
                "status": "completed",
                ...
            },
            ...
        ],
        
        "manual_interventions": [
            {
                "timestamp": "2026-01-15T10:45:00Z",
                "step_id": "03_binning",
                "action": "bin_override",
                "variable": "age",
                "reason": "Business rule: age < 18 must be separate bin",
                "before": {"bins": [1, 2, 3]},
                "after": {"bins": [1, 2, 2, 3]},
                "performed_by": "modeler_user_id",
                "audit_log_ref": "audit_20260115_001_123"
            }
        ],
        
        "summary_statistics": {
            "total_execution_time_seconds": 8745.2,
            "peak_memory_gb": 12.3,
            "spark_executors_used": 16,
            "data_compression_ratio": 18.5
        },
        
        "errors": [],  // Empty if no errors
        "warnings": [
            {
                "step_id": "05_model",
                "level": "warning",
                "message": "Multicollinearity detected in features: income_annual, income_monthly",
                "recommended_action": "Review feature selection"
            }
        ],
        
        "validation_results": {
            "reproducibility_test": "passed",
            "artifact_integrity": "passed",
            "schema_validation": "passed"
        }
    }

================================================================================
3. EDA ARTIFACTS
================================================================================

eda_summary.json:

    {
        "run_id": "run_20260115_001",
        "step_id": "01_eda",
        "segments": {
            "overall": {
                "row_count": 5000000,
                "event_count": 1234567,
                "event_rate": 0.247314,
                "missing_value_summary": {
                    "age": {"count": 1234, "pct": 0.0002},
                    "income": {"count": 5678, "pct": 0.0011}
                }
            },
            "product_A": {
                "row_count": 2000000,
                "event_count": 450000,
                "event_rate": 0.225,
                ...
            }
        },
        "variables": {
            "age": {
                "type": "numeric",
                "min": 18,
                "max": 90,
                "mean": 45.2,
                "std": 12.3,
                "median": 44,
                "percentiles": {"1": 20, "5": 25, "25": 35, "75": 55, "95": 70, "99": 80},
                "unique_count": 72,
                "missing_count": 1234,
                "distribution": {
                    "bins": [10, 20, 30, 40, 50, 60, 70, 80],
                    "counts": [120, 450, 890, 1200, 950, 670, 320, 100]
                }
            },
            "product": {
                "type": "categorical",
                "unique_values": ["A", "B", "C", "D"],
                "value_counts": {"A": 2000000, "B": 1500000, "C": 1000000, "D": 500000},
                "missing_count": 0
            }
        },
        "drift_metrics": {
            "psi": {
                "age": {"value": 0.023, "status": "low_drift"},
                "income": {"value": 0.089, "status": "medium_drift"}
            },
            "csi": {
                ...
            }
        }
    }

eda_univariate.csv:

    segment,variable,type,count,missing_count,missing_pct,unique_count,\
    min,max,mean,std,median,q1,q25,q75,q95,q99,event_rate
    overall,age,numeric,5000000,1234,0.0002,72,18,90,45.2,12.3,44,22,35,55,70,80,0.2473
    overall,income,numeric,5000000,5678,0.0011,8945,5000,500000,85234,78234,72000,40000,95000,180000,350000,450000,0.2467
    product_A,age,numeric,2000000,500,0.0003,72,18,90,44.1,11.8,43,21,34,54,69,79,0.2250
    ...

eda_bivariate.csv:

    variable_1,variable_2,correlation,chi_square_stat,pvalue,notes
    age,income,0.456,12345.2,<0.001,Strong positive correlation
    age,default,0.234,8934.1,<0.001,Moderate positive (event correlation)
    income,default,0.189,5623.8,<0.001,Weak positive
    ...

================================================================================
4. BINNING ARTIFACTS
================================================================================

binning_tables/{variable}.json (one per variable):

    {
        "run_id": "run_20260115_001",
        "step_id": "03_binning",
        "variable_name": "age",
        "variable_type": "numeric",
        "bins": [
            {
                "bin_id": 1,
                "label": "18-25",
                "lower_bound": 18,
                "upper_bound": 25,
                "count": 450000,
                "event_count": 95000,
                "event_rate": 0.2111,
                "non_event_count": 355000,
                "woe": -0.341,
                "iv": 0.0123,
                "pct_of_total": 0.0900
            },
            {
                "bin_id": 2,
                "label": "26-35",
                "lower_bound": 26,
                "upper_bound": 35,
                "count": 890000,
                "event_count": 223000,
                "event_rate": 0.2506,
                "non_event_count": 667000,
                "woe": -0.0145,
                "iv": 0.0002,
                "pct_of_total": 0.1780
            },
            ...
            {
                "bin_id": "MISSING",
                "label": "Missing",
                "count": 1234,
                "event_count": 280,
                "event_rate": 0.2270,
                "woe": -0.0892,
                "iv": 0.0001,
                "pct_of_total": 0.0002
            }
        ],
        "summary": {
            "total_count": 5000000,
            "total_events": 1234567,
            "total_iv": 0.1567,
            "monotonic_event_rate": true,
            "num_bins": 12,
            "missing_count": 1234,
            "enforcement_applied": "monotonic_merge"
        },
        "overrides": [
            {
                "timestamp": "2026-01-15T10:45:00Z",
                "performed_by": "modeler_user_id",
                "reason": "Business rule: age < 18 must be separate bin",
                "before_merge": [[18, 25], [26, 35]],
                "after_merge": [[18, 18], [19, 25], [26, 35]],
                "audit_log_ref": "audit_20260115_001_123"
            }
        ]
    }

binning_summary.csv (cross-variable overview):

    variable,type,bins_created,iv,monotonic,missing_handling
    age,numeric,12,0.1567,TRUE,separate_bin
    income,numeric,15,0.2345,TRUE,separate_bin
    product,categorical,4,0.0890,FALSE,separate_category
    ...

================================================================================
5. WOE MAPPING ARTIFACTS
================================================================================

woe_mappings/{variable}.json:

    {
        "run_id": "run_20260115_001",
        "step_id": "04_encoding",
        "variable_name": "age",
        "encoding_type": "woe",
        "mappings": [
            {
                "bin_id": 1,
                "bin_label": "18-25",
                "input_values": {"numeric_range": [18, 25]},
                "woe_value": -0.341,
                "encoding_type": "numeric"
            },
            {
                "bin_id": 2,
                "bin_label": "26-35",
                "input_values": {"numeric_range": [26, 35]},
                "woe_value": -0.0145,
                "encoding_type": "numeric"
            },
            {
                "bin_id": "MISSING",
                "bin_label": "Missing",
                "input_values": {"pattern": "null"},
                "woe_value": -0.0892,
                "encoding_type": "missing"
            }
        ],
        "global_event_rate": 0.2473,
        "validation": {
            "encoding_reversible": true,
            "all_bins_mapped": true
        }
    }

scoring_mappings.json (for operational scoring):

    {
        "run_id": "run_20260115_001",
        "step_id": "07_export",
        "scoring_type": "points",
        "variables": {
            "age": {
                "variable_type": "numeric",
                "mappings": [
                    {"condition": "age < 18", "points": 25},
                    {"condition": "age >= 18 AND age < 25", "points": 15},
                    {"condition": "age >= 25 AND age < 35", "points": 20},
                    ...
                ]
            },
            "income": {
                "variable_type": "numeric",
                "mappings": [
                    {"condition": "income < 30000", "points": 5},
                    ...
                ]
            }
        }
    }

================================================================================
6. COMPRESSION ARTIFACTS
================================================================================

compression_summary.json:

    {
        "run_id": "run_20260115_001",
        "step_id": "02_compression",
        "compression_mode": "post_binning_exact",
        "input_statistics": {
            "row_count": 5000000,
            "event_count": 1234567,
            "event_rate": 0.2473,
            "size_mb": 2500
        },
        "output_statistics": {
            "row_count": 125000,
            "event_count": 1234567,
            "event_rate": 0.2473,
            "size_mb": 45
        },
        "compression_ratio": 55.56,
        "verification": {
            "total_rows_preserved": true,
            "total_events_preserved": true,
            "event_rate_preserved": true,
            "likelihood_preserved": true,
            "verification_status": "passed"
        },
        "weight_contract": {
            "sample_weight_total": 5000000,
            "event_weight_total": 1234567,
            "sample_weight_type": "count",
            "event_weight_type": "count"
        }
    }

weight_contract.json (operational reference):

    {
        "version": "1.0",
        "run_id": "run_20260115_001",
        "created_timestamp": "2026-01-15T10:00:00Z",
        "semantics": {
            "sample_weight": "number of rows represented",
            "event_weight": "number of events represented"
        },
        "guarantees": {
            "total_observations_preserved": true,
            "total_events_preserved": true,
            "event_rates_preserved": true,
            "logistic_likelihood_preserved": true
        },
        "expected_values": {
            "total_sample_weight": 5000000,
            "total_event_weight": 1234567,
            "global_event_rate": 0.2473
        }
    }

================================================================================
7. MODEL ARTIFACTS
================================================================================

model_coefficients.csv:

    variable,coefficient,std_error,z_stat,pvalue,sig_level,iv_contribution
    age_25,0.234,0.012,19.5,<0.001,***,0.0234
    age_35,0.156,0.011,14.2,<0.001,***,0.0145
    income_1,-0.456,0.015,-30.4,<0.001,***,0.0678
    income_2,-0.234,0.014,-16.7,<0.001,***,0.0234
    (Intercept),-1.234,0.025,-49.3,<0.001,***,N/A

model_metrics.json:

    {
        "run_id": "run_20260115_001",
        "step_id": "05_model",
        "model_type": "logistic_regression",
        "feature_count": 45,
        "training_rows": 3000000,
        "validation_rows": 2000000,
        "coefficients_file": "model_coefficients.csv",
        "performance_metrics": {
            "training": {
                "auc_roc": 0.756,
                "gini": 0.512,
                "ks_statistic": 0.345,
                "log_loss": 0.523
            },
            "validation": {
                "auc_roc": 0.743,
                "gini": 0.486,
                "ks_statistic": 0.328,
                "log_loss": 0.534
            }
        },
        "stability_metrics": {
            "psi_train_to_val": 0.023,
            "coefficient_stability": 0.987
        }
    }

================================================================================
8. SCALING & SCORING ARTIFACTS
================================================================================

points_table.csv:

    variable,bin,points,odds_contribution
    age,18-25,15,0.34
    age,26-35,20,0.42
    age,36-50,25,0.51
    age,51+,30,0.61
    income,<30K,5,0.12
    income,30-50K,10,0.24
    income,50-100K,15,0.37
    income,100K+,20,0.49

score_formula.json:

    {
        "run_id": "run_20260115_001",
        "step_id": "06_scaling",
        "scaling_method": "pdo_based",
        "parameters": {
            "pdo": 50,
            "base_score": 600,
            "base_odds": 1.0,
            "odds_multiplier": 2.0
        },
        "formula": {
            "description": "Score = base_score + (pdo / ln(2)) * log(odds)",
            "score_range": [300, 900],
            "interpretation": "Higher score = lower default risk"
        },
        "score_bands": [
            {
                "min_score": 300,
                "max_score": 450,
                "label": "Very High Risk",
                "approval_probability": 0.10
            },
            {
                "min_score": 450,
                "max_score": 550,
                "label": "High Risk",
                "approval_probability": 0.30
            },
            {
                "min_score": 550,
                "max_score": 650,
                "label": "Medium Risk",
                "approval_probability": 0.60
            },
            {
                "min_score": 650,
                "max_score": 750,
                "label": "Low Risk",
                "approval_probability": 0.80
            },
            {
                "min_score": 750,
                "max_score": 900,
                "label": "Very Low Risk",
                "approval_probability": 0.95
            }
        ]
    }

================================================================================
9. FINAL ARTIFACTS & EXPORT
================================================================================

final_report.html:
    - Auto-generated HTML with all plots, tables, summary statistics
    - Standalone file (all CSS/JS embedded)
    - Sections:
      - Executive Summary
      - Data Overview
      - Feature Analysis
      - Binning Summary
      - Model Performance
      - Scoring Algorithm
      - Monitoring Plan

model_card.md:
    - Machine Learning Model Card (standardized format)
    - Sections:
      - Model Details (type, version, date)
      - Target & Features
      - Training Data (size, characteristics, fairness analysis)
      - Evaluation (metrics, slices, limitations)
      - Use Cases (intended, not intended)
      - Ethical Considerations
      - Monitoring (vintage curves, PSI tracking)

scoring_spec.json:
    - Operational scoring specification for deployment
    - Includes:
      - Feature engineering rules
      - Binning logic
      - Scoring formula
      - Points table
      - Reason code logic
      - Score range and interpretation

================================================================================
10. ARTIFACT COMPARISON SPECIFICATION
================================================================================

Comparison Report (comparing two runs):

    {
        "run_a_id": "run_20260115_001",
        "run_b_id": "run_20260115_002",
        "comparison_timestamp": "2026-01-15T14:00:00Z",
        "summary": {
            "config_changed": true,
            "data_changed": false,
            "results_changed": true,
            "overall_status": "acceptable_differences"
        },
        "config_diff": {
            "differences": [
                {
                    "field": "binning.method",
                    "value_a": "equal_width",
                    "value_b": "quantile"
                }
            ]
        },
        "model_metrics_diff": {
            "auc_roc_a": 0.756,
            "auc_roc_b": 0.758,
            "auc_roc_delta": 0.002,
            "auc_roc_delta_pct": 0.26,
            "status": "acceptable"
        },
        "scoring_diff": {
            "score_range_a": [300, 900],
            "score_range_b": [300, 900],
            "percentile_differences": [
                {"percentile": 50, "score_a": 650, "score_b": 652, "delta": 2},
                {"percentile": 90, "score_a": 780, "score_b": 785, "delta": 5}
            ]
        }
    }

================================================================================
