"""
CR_Score: Enterprise Scorecard Development Platform

A config-driven, Spark-native platform for end-to-end credit scorecard development.

Key Features:
    - Config-first, artifact-first, deterministic-by-default design
    - Spark-based large-scale processing with intelligent compression
    - Full scorecard lifecycle support
    - Multiple interfaces (CLI, SDK, API, UI)
    - MCP/Tool integration for agent workflows
    - Enterprise-grade audit logging and reproducibility

Example:
    >>> from cr_score import ScorecardPipeline
    >>> pipeline = ScorecardPipeline()
    >>> results = pipeline.fit(df_train, target_col="default")
    >>> scores = pipeline.predict(df_test)
"""

__version__ = "1.2.0"
__author__ = "Edmun Lee"

from cr_score.core.config.schema import Config
from cr_score.core.registry.run_registry import RunRegistry
from cr_score.core.exceptions import CR_ScoreException
from cr_score.pipeline import ScorecardPipeline

__all__ = [
    "__version__",
    "__author__",
    "Config",
    "RunRegistry",
    "CR_ScoreException",
    "ScorecardPipeline",
]
