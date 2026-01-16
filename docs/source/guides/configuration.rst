Configuration Guide
===================

CR_Score uses YAML configuration files for all settings.

Basic Configuration
-------------------

See ``src/cr_score/templates/intermediate/config_template.yml`` for a complete example.

Pipeline Configuration
----------------------

.. code-block:: python

   from cr_score import ScorecardPipeline

   pipeline = ScorecardPipeline(
       max_n_bins=5,           # Maximum bins per feature
       min_iv=0.02,            # Minimum IV to include feature
       pdo=20,                 # Points to double odds
       base_score=600,         # Score at base odds
       base_odds=50.0,         # Odds at base score (2% default rate)
       target_bad_rate=0.05,   # Target bad rate for calibration
       calibrate=True,         # Enable calibration
       feature_selection="stepwise",  # Feature selection method
       max_features=10,        # Maximum features to select
       random_state=42         # Random seed
   )

For more details, see :doc:`/api/pipeline`.
