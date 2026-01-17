CR_Score Documentation
======================

**CR_Score** is an enterprise-grade, config-driven platform for end-to-end credit scorecard development with deterministic reproducibility and enterprise-grade audit trails.

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: https://github.com/edmunlee87/CR_Score
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue
   :target: https://www.python.org/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/edmunlee87/CR_Score/blob/main/LICENSE
   :alt: License

Key Features
------------

- **Config-First Design**: Every action expressible as YAML configuration
- **Artifact-First**: All outputs versioned, hashed, and auditable
- **Deterministic by Default**: Same config + same data = same result
- **Spark-Native**: Efficient processing of 100M+ rows with intelligent compression
- **Model-Agnostic Feature Selection**: Forward, backward, stepwise, exhaustive with MLflow tracking
- **Automated Binning**: OptBinning integration for optimal WoE transformation
- **Simplified 3-Line Interface**: Build complete scorecards with minimal code
- **Production Monitoring & Observability**: Performance tracking, drift detection, alerting
- **SHAP Explainability**: Model explanations and regulatory-compliant reason codes
- **Interactive Visualizations**: Plotly-based charts for binning, scores, performance
- **HTML Report Generation**: Professional reports with embedded visualizations
- **Observability Dashboards**: Real-time production monitoring dashboards
- **MCP Tools**: Agent-ready workflows for AI integration
- **Enterprise Audit Trails**: Structured logging for compliance

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/edmunlee87/CR_Score.git
   cd CR_Score
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"

3-Line Scorecard
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline

   pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
   pipeline.fit(df_train, target_col="default")
   scores = pipeline.predict(df_test)

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/installation
   guides/quickstart
   guides/configuration
   guides/feature_selection
   guides/enhanced_features
   guides/visualization
   guides/reporting
   guides/explainability
   guides/observability
   guides/spark_optimization
   guides/mcp_tools

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/simple_scorecard
   examples/complete_workflow
   examples/feature_selection_mlflow
   examples/custom_binning

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/core
   api/data
   api/spark
   api/eda
   api/binning
   api/encoding
   api/features
   api/model
   api/calibration
   api/scaling
   api/pipeline
   api/explainability
   api/monitoring
   api/viz
   api/reporting
   api/templates
   api/tools

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Architecture
------------

CR_Score follows a modular, config-driven architecture:

.. code-block:: text

   Data → EDA → Feature Engineering → Binning → WoE Encoding →
   Reject Inference → Modeling → Calibration → Scaling → Reporting → Export

Core Principles
~~~~~~~~~~~~~~~

1. **Config-First**: No hardcoded values, all defaults overridable
2. **Artifact-First**: Every step produces versioned artifacts
3. **Deterministic**: Reproducible results with fixed seeds and hashing
4. **Spark Where It Matters**: Heavy operations in Spark, orchestration in Python
5. **Scale Without Losing Correctness**: Post-binning compression reduces data 20x-100x

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
