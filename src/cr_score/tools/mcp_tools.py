"""
MCP (Model Context Protocol) tools for agent integration.

Provides standardized tool definitions for AI agents to interact with CR_Score.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from cr_score import ScorecardPipeline
from cr_score.features import ForwardSelector, BackwardSelector, StepwiseSelector
from cr_score.core.logging import get_audit_logger


logger = get_audit_logger()


def score_predict_tool(
    data_path: str,
    model_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict credit scores for new data using trained model.

    Args:
        data_path: Path to input data (CSV/Parquet)
        model_path: Path to trained pipeline model
        output_path: Path to save predictions (optional)

    Returns:
        Dictionary with predictions and summary statistics

    MCP Tool Specification:
        name: score_predict
        description: Predict credit scores for new customers
        input_schema:
            data_path: string (required) - Input data file path
            model_path: string (required) - Trained model file path
            output_path: string (optional) - Output predictions path
        
    Example:
        >>> result = score_predict_tool(
        ...     data_path="new_applications.csv",
        ...     model_path="models/scorecard_v1.pkl"
        ... )
        >>> print(f"Scored {result['n_records']} records")
    """
    logger.info("score_predict_tool called", data_path=data_path, model_path=model_path)

    try:
        # Load data
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Load model
        import pickle
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        # Predict
        scores = pipeline.predict(df)
        probas = pipeline.predict_proba(df)

        # Create output
        df_output = df.copy()
        df_output["credit_score"] = scores
        df_output["default_probability"] = probas

        # Save if requested
        if output_path:
            df_output.to_csv(output_path, index=False)

        # Return summary
        return {
            "status": "success",
            "n_records": len(df),
            "score_statistics": {
                "mean": float(scores.mean()),
                "median": float(pd.Series(scores).median()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "std": float(scores.std()),
            },
            "probability_statistics": {
                "mean": float(probas.mean()),
                "median": float(pd.Series(probas).median()),
            },
            "output_path": output_path,
        }

    except Exception as e:
        logger.error("score_predict_tool failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def model_evaluate_tool(
    data_path: str,
    model_path: str,
    target_col: str = "default",
) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.

    Args:
        data_path: Path to test data with ground truth
        model_path: Path to trained pipeline model
        target_col: Name of target column

    Returns:
        Dictionary with performance metrics

    MCP Tool Specification:
        name: model_evaluate
        description: Evaluate scorecard model performance
        input_schema:
            data_path: string (required) - Test data file path
            model_path: string (required) - Trained model file path
            target_col: string (optional) - Target column name
        
    Example:
        >>> result = model_evaluate_tool(
        ...     data_path="test_data.csv",
        ...     model_path="models/scorecard_v1.pkl"
        ... )
        >>> print(f"AUC: {result['metrics']['auc']:.3f}")
    """
    logger.info("model_evaluate_tool called", data_path=data_path, model_path=model_path)

    try:
        # Load data
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Load model
        import pickle
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

        # Evaluate
        metrics = pipeline.evaluate(df, target_col=target_col)

        # Add interpretation
        interpretation = {
            "auc": "Excellent" if metrics["auc"] >= 0.8 else "Good" if metrics["auc"] >= 0.7 else "Fair",
            "gini": f"Discrimination power: {metrics['gini']:.2%}",
            "ks": f"Max separation: {metrics['ks']:.2%}",
        }

        return {
            "status": "success",
            "metrics": metrics,
            "interpretation": interpretation,
            "n_records": len(df),
        }

    except Exception as e:
        logger.error("model_evaluate_tool failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def feature_select_tool(
    data_path: str,
    target_col: str,
    method: str = "stepwise",
    max_features: int = 10,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Select best features using automated feature selection.

    Args:
        data_path: Path to training data
        target_col: Name of target column
        method: Selection method (forward, backward, stepwise)
        max_features: Maximum features to select
        output_path: Path to save selected features (optional)

    Returns:
        Dictionary with selected features and scores

    MCP Tool Specification:
        name: feature_select
        description: Automatically select best features for scorecard
        input_schema:
            data_path: string (required) - Training data file path
            target_col: string (required) - Target column name
            method: string (optional) - forward/backward/stepwise
            max_features: integer (optional) - Max features to select
            output_path: string (optional) - Output path for results
        
    Example:
        >>> result = feature_select_tool(
        ...     data_path="train_data.csv",
        ...     target_col="default",
        ...     method="stepwise",
        ...     max_features=10
        ... )
        >>> print(f"Selected {len(result['selected_features'])} features")
    """
    logger.info(
        "feature_select_tool called",
        data_path=data_path,
        method=method,
        max_features=max_features,
    )

    try:
        # Load data
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Select method
        from sklearn.linear_model import LogisticRegression

        if method == "forward":
            selector = ForwardSelector(
                estimator=LogisticRegression(random_state=42, max_iter=1000),
                max_features=max_features,
                use_mlflow=False,
            )
        elif method == "backward":
            selector = BackwardSelector(
                estimator=LogisticRegression(random_state=42, max_iter=1000),
                min_features=max(1, max_features // 2),
                use_mlflow=False,
            )
        elif method == "stepwise":
            selector = StepwiseSelector(
                estimator=LogisticRegression(random_state=42, max_iter=1000),
                max_features=max_features,
                use_mlflow=False,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fit selector
        selector.fit(X, y)

        selected_features = selector.get_selected_features()
        importance = selector.get_feature_importance()

        # Save if requested
        if output_path:
            pd.DataFrame({"feature": selected_features}).to_csv(output_path, index=False)

        return {
            "status": "success",
            "method": method,
            "n_features_total": X.shape[1],
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "best_score": float(selector.best_score_),
            "output_path": output_path,
        }

    except Exception as e:
        logger.error("feature_select_tool failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def binning_analyze_tool(
    data_path: str,
    feature_col: str,
    target_col: str,
    max_bins: int = 10,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze optimal binning for a feature.

    Args:
        data_path: Path to data
        feature_col: Feature to bin
        target_col: Target column
        max_bins: Maximum bins
        output_path: Path to save binning table (optional)

    Returns:
        Dictionary with binning analysis

    MCP Tool Specification:
        name: binning_analyze
        description: Analyze optimal binning for scorecard features
        input_schema:
            data_path: string (required) - Data file path
            feature_col: string (required) - Feature to analyze
            target_col: string (required) - Target column name
            max_bins: integer (optional) - Maximum bins
            output_path: string (optional) - Output path for results
        
    Example:
        >>> result = binning_analyze_tool(
        ...     data_path="train_data.csv",
        ...     feature_col="age",
        ...     target_col="default",
        ...     max_bins=5
        ... )
        >>> print(f"IV: {result['iv']:.3f}")
    """
    logger.info(
        "binning_analyze_tool called",
        data_path=data_path,
        feature_col=feature_col,
        max_bins=max_bins,
    )

    try:
        # Load data
        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Use OptBinning
        from cr_score.binning import OptBinningWrapper

        binner = OptBinningWrapper(max_n_bins=max_bins)
        binner.fit(df[[feature_col]], df[target_col], features=[feature_col])

        binning_table = binner.get_binning_table(feature_col)
        iv = binner.get_iv(feature_col)

        # Save if requested
        if output_path:
            binning_table.to_csv(output_path, index=False)

        # Interpret IV
        if iv < 0.02:
            iv_strength = "Weak"
        elif iv < 0.1:
            iv_strength = "Medium"
        elif iv < 0.3:
            iv_strength = "Strong"
        else:
            iv_strength = "Very Strong"

        return {
            "status": "success",
            "feature": feature_col,
            "n_bins": len(binning_table),
            "iv": float(iv),
            "iv_strength": iv_strength,
            "binning_table": binning_table.to_dict("records"),
            "output_path": output_path,
        }

    except Exception as e:
        logger.error("binning_analyze_tool failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


# Tool Registry for MCP Server
MCP_TOOLS = {
    "score_predict": {
        "function": score_predict_tool,
        "name": "score_predict",
        "description": "Predict credit scores for new customers using trained scorecard model",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to input data file (CSV or Parquet)",
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to trained scorecard model file",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save predictions",
                },
            },
            "required": ["data_path", "model_path"],
        },
    },
    "model_evaluate": {
        "function": model_evaluate_tool,
        "name": "model_evaluate",
        "description": "Evaluate scorecard model performance on test data",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to test data file with ground truth",
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to trained scorecard model file",
                },
                "target_col": {
                    "type": "string",
                    "description": "Name of target column (default: 'default')",
                },
            },
            "required": ["data_path", "model_path"],
        },
    },
    "feature_select": {
        "function": feature_select_tool,
        "name": "feature_select",
        "description": "Automatically select best features for scorecard using forward/backward/stepwise selection",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to training data file",
                },
                "target_col": {
                    "type": "string",
                    "description": "Name of target column",
                },
                "method": {
                    "type": "string",
                    "enum": ["forward", "backward", "stepwise"],
                    "description": "Feature selection method (default: stepwise)",
                },
                "max_features": {
                    "type": "integer",
                    "description": "Maximum number of features to select (default: 10)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save selected features",
                },
            },
            "required": ["data_path", "target_col"],
        },
    },
    "binning_analyze": {
        "function": binning_analyze_tool,
        "name": "binning_analyze",
        "description": "Analyze optimal binning for scorecard features with IV calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_path": {
                    "type": "string",
                    "description": "Path to data file",
                },
                "feature_col": {
                    "type": "string",
                    "description": "Feature column to analyze",
                },
                "target_col": {
                    "type": "string",
                    "description": "Target column name",
                },
                "max_bins": {
                    "type": "integer",
                    "description": "Maximum number of bins (default: 10)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save binning table",
                },
            },
            "required": ["data_path", "feature_col", "target_col"],
        },
    },
}
