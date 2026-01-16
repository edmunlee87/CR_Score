"""
Verification script for CR_Score v1.0.0

Quick test to verify all major components are working.
"""

import sys


def verify_imports():
    """Verify all major components can be imported."""
    print("=" * 80)
    print("CR_Score v1.0.0 - Installation Verification")
    print("=" * 80)
    print()
    
    components = [
        ("Core - Config", "cr_score.core.config.schema", "ProjectConfig"),
        ("Core - Logging", "cr_score.core.logging", "get_audit_logger"),
        ("Core - Registry", "cr_score.core.registry", "RunRegistry"),
        ("Data - Connectors", "cr_score.data.connectors", "LocalFileConnector"),
        ("Data - Validation", "cr_score.data.validation", "SchemaChecker"),
        ("Spark - Session", "cr_score.spark.session", "SparkSessionFactory"),
        ("EDA - Univariate", "cr_score.eda.univariate", "UnivariateAnalyzer"),
        ("EDA - Bivariate", "cr_score.eda.bivariate", "BivariateAnalyzer"),
        ("Binning - OptBinning", "cr_score.binning", "OptBinningWrapper"),
        ("Encoding - WoE", "cr_score.encoding.woe", "WoEEncoder"),
        ("Reject Inference", "cr_score.reject_inference", "ParcelingRejectInference"),
        ("Model - Logistic", "cr_score.model.logistic", "LogisticScorecard"),
        ("Calibration", "cr_score.calibration", "InterceptCalibrator"),
        ("Scaling - PDO", "cr_score.scaling.pdo", "PDOScaler"),
        ("Features - Selection", "cr_score.features", "ForwardSelector"),
        ("Features - Selection", "cr_score.features", "StepwiseSelector"),
        ("Visualization - Binning", "cr_score.viz", "BinningVisualizer"),
        ("Visualization - Score", "cr_score.viz", "ScoreVisualizer"),
        ("Reporting", "cr_score.reporting", "HTMLReportGenerator"),
        ("MCP Tools", "cr_score.tools", "score_predict_tool"),
        ("Pipeline", "cr_score", "ScorecardPipeline"),
    ]
    
    failed = []
    
    for component_name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {component_name:30s} - OK")
        except ImportError as e:
            print(f"✗ {component_name:30s} - FAILED (ImportError: {e})")
            failed.append(component_name)
        except AttributeError as e:
            print(f"✗ {component_name:30s} - FAILED (AttributeError: {e})")
            failed.append(component_name)
        except Exception as e:
            print(f"✗ {component_name:30s} - FAILED ({type(e).__name__}: {e})")
            failed.append(component_name)
    
    print()
    print("=" * 80)
    
    if failed:
        print(f"❌ Verification FAILED - {len(failed)} component(s) failed:")
        for component in failed:
            print(f"   - {component}")
        print()
        print("Please check your installation and dependencies.")
        return False
    else:
        print(f"✅ Verification PASSED - All {len(components)} components working!")
        print()
        print("CR_Score v1.0.0 is ready to use!")
        print()
        print("Quick Start:")
        print("  from cr_score import ScorecardPipeline")
        print("  pipeline = ScorecardPipeline()")
        print("  pipeline.fit(df_train, target_col='default')")
        print("  scores = pipeline.predict(df_test)")
        print()
        print("Examples:")
        print("  python examples/simple_3_line_scorecard.py")
        print("  python examples/complete_scorecard_workflow.py")
        print("  python examples/feature_selection_with_mlflow.py")
        print()
        return True


def main():
    """Run verification."""
    success = verify_imports()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
