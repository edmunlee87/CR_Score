Quick Start Guide
=================

This guide will get you up and running with CR_Score in 5 minutes.

Installation
------------

.. code-block:: bash

   git clone https://github.com/edmunlee87/CR_Score.git
   cd CR_Score
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"

Verify Installation
-------------------

.. code-block:: bash

   python verify_installation.py

You should see all components marked with ✓.

Your First Scorecard
--------------------

Build a complete scorecard in 3 lines:

.. code-block:: python

   from cr_score import ScorecardPipeline
   import pandas as pd

   # Load your data
   df = pd.read_csv("applications.csv")

   # Split train/test
   train = df.iloc[:int(len(df)*0.7)]
   test = df.iloc[int(len(df)*0.7):]

   # Build scorecard in 3 lines
   pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
   pipeline.fit(train, target_col="default")
   scores = pipeline.predict(test)

   # Evaluate
   metrics = pipeline.evaluate(test, target_col="default")
   print(f"AUC: {metrics['auc']:.3f}")
   print(f"Gini: {metrics['gini']:.3f}")
   print(f"KS: {metrics['ks']:.3f}")

Understanding the Results
--------------------------

Scores
~~~~~~

The ``predict`` method returns credit scores scaled using PDO:

- Higher scores = lower risk
- Typical range: 300-850
- Every 20 points (PDO), odds of default double

Probabilities
~~~~~~~~~~~~~

The ``predict_proba`` method returns default probabilities:

.. code-block:: python

   probas = pipeline.predict_proba(test)
   # probas in range [0, 1]

Metrics
~~~~~~~

The ``evaluate`` method returns:

- **AUC**: Area Under ROC Curve (0.5-1.0, higher is better)
  
  - >= 0.8: Excellent
  - >= 0.7: Good
  - >= 0.6: Fair

- **Gini**: 2 * AUC - 1, measures discrimination power
- **KS**: Kolmogorov-Smirnov statistic, max separation between good/bad
- **Accuracy, Precision, Recall, F1**: Standard classification metrics

Next Steps
----------

- :doc:`/guides/feature_selection` - Select best features automatically
- :doc:`/guides/visualization` - Create interactive charts
- :doc:`/guides/reporting` - Generate HTML reports
- :doc:`/examples/complete_workflow` - See complete workflow

Run Examples
------------

Try the included examples:

.. code-block:: bash

   # Simple 3-line scorecard
   python examples/simple_3_line_scorecard.py

   # Complete workflow
   python examples/complete_scorecard_workflow.py

   # Feature selection with MLflow
   python examples/feature_selection_with_mlflow.py

Get Help
--------

- GitHub Issues: https://github.com/edmunlee87/CR_Score/issues
- API Documentation: :doc:`/api/pipeline`
- Examples: :doc:`/examples/simple_scorecard`
