Configuration Templates
========================

Ready-to-use configuration templates for different use cases and skill levels.

Overview
--------

The templates module provides pre-configured YAML templates for:

- **Beginner**: Simple scorecards with minimal configuration
- **Intermediate**: Full-featured scorecards with EDA and feature selection
- **Advanced**: Enterprise-grade with monitoring and observability

Template Levels
---------------

Beginner Templates
~~~~~~~~~~~~~~~~~~

**For:** First-time users, simple scorecards, prototyping  
**Features:** Minimal configuration, optimal defaults  
**Files:** ``basic_scorecard.yml``

.. code-block:: yaml

   # Example beginner template
   project:
     name: "my_first_scorecard"
     description: "Basic credit scorecard"
   
   data:
     train_path: "data/train.csv"
     test_path: "data/test.csv"
     target_column: "default"
   
   binning:
     method: "optbinning"
     max_n_bins: 5
   
   model:
     type: "logistic"
   
   scaling:
     pdo: 20
     base_score: 600
     base_odds: 50

Intermediate Templates
~~~~~~~~~~~~~~~~~~~~~~

**For:** Production-ready scorecards  
**Features:** Full EDA, feature selection, comprehensive reporting  
**Files:** ``full_scorecard.yml``

**Additional sections:**
- ``eda``: Exploratory data analysis
- ``features.selection``: Feature selection methods
- ``calibration``: Model calibration
- ``viz``: Visualization settings

Advanced Templates
~~~~~~~~~~~~~~~~~~

**For:** Enterprise deployments with full MLOps  
**Features:** Monitoring, observability, compliance, governance  
**Files:** ``production_scorecard.yml``

**Additional sections:**
- ``monitoring.performance``: Performance tracking
- ``monitoring.drift``: Drift detection
- ``monitoring.predictions``: Prediction monitoring
- ``monitoring.alerts``: Alert configuration
- ``monitoring.metrics``: Metrics collection
- ``explainability``: Model explainability
- ``governance``: Compliance and audit
- ``interfaces``: API/UI configuration

Template API
------------

.. automodule:: cr_score.templates
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Loading Templates
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.templates import get_template_path, list_templates
   import yaml
   
   # List all available templates
   templates = list_templates()
   print(templates)
   # {'beginner': ['basic_scorecard.yml'],
   #  'intermediate': ['full_scorecard.yml'],
   #  'advanced': ['production_scorecard.yml']}
   
   # Get template path
   template_path = get_template_path('beginner', 'basic_scorecard.yml')
   
   # Load template
   with open(template_path) as f:
       config = yaml.safe_load(f)

Customizing Templates
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import yaml
   from cr_score.templates import get_template_path
   
   # Load beginner template
   template_path = get_template_path('beginner', 'basic_scorecard.yml')
   with open(template_path) as f:
       config = yaml.safe_load(f)
   
   # Customize
   config['project']['name'] = 'my_custom_scorecard'
   config['data']['train_path'] = 'my_data/train.csv'
   config['data']['test_path'] = 'my_data/test.csv'
   config['scaling']['pdo'] = 50  # Custom PDO
   
   # Save custom config
   with open('my_config.yml', 'w') as f:
       yaml.dump(config, f)

Using with Pipeline
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.templates import get_template_path
   import yaml
   
   # Load and customize template
   template_path = get_template_path('intermediate', 'full_scorecard.yml')
   with open(template_path) as f:
       config = yaml.safe_load(f)
   
   config['data']['train_path'] = 'my_data.csv'
   
   # Create pipeline from config
   pipeline = ScorecardPipeline.from_config(config)
   
   # Or manually set parameters
   pipeline = ScorecardPipeline(
       max_n_bins=config['binning']['max_n_bins'],
       pdo=config['scaling']['pdo'],
       base_score=config['scaling']['base_score']
   )

Template Comparison
-------------------

.. list-table:: Feature Comparison
   :header-rows: 1
   :widths: 40 20 20 20

   * - Feature
     - Beginner
     - Intermediate
     - Advanced
   * - Basic Scorecard
     - ✅
     - ✅
     - ✅
   * - EDA
     - ❌
     - ✅
     - ✅
   * - Feature Selection
     - ❌
     - ✅
     - ✅
   * - Optimal Binning
     - ✅
     - ✅
     - ✅
   * - Calibration
     - ❌
     - ✅
     - ✅
   * - Performance Monitoring
     - ❌
     - ❌
     - ✅
   * - Drift Detection
     - ❌
     - ❌
     - ✅
   * - Explainability
     - ❌
     - ❌
     - ✅
   * - Alert Management
     - ❌
     - ❌
     - ✅
   * - Metrics Collection
     - ❌
     - ❌
     - ✅
   * - Observability Dashboard
     - ❌
     - ❌
     - ✅

Configuration Sections
----------------------

Core Sections (All Levels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**project**
   Project metadata (name, description, version)

**data**
   Data sources and paths

**binning**
   Binning configuration (method, bins, constraints)

**model**
   Model type and hyperparameters

**scaling**
   PDO scaling parameters

Intermediate Sections
~~~~~~~~~~~~~~~~~~~~~

**eda**
   Exploratory data analysis settings

**features**
   Feature engineering and selection

**calibration**
   Model calibration settings

**viz**
   Visualization preferences

Advanced Sections
~~~~~~~~~~~~~~~~~

**monitoring**
   Complete monitoring configuration:
   
   - ``performance``: Performance tracking
   - ``drift``: Drift detection settings
   - ``predictions``: Prediction monitoring
   - ``alerts``: Alert thresholds and routing
   - ``metrics``: Metrics collection

**explainability**
   Model explainability settings:
   
   - ``shap``: SHAP configuration
   - ``reason_codes``: Reason code generation
   - ``feature_importance``: Importance methods

**governance**
   Compliance and audit:
   
   - ``model_card``: Model card generation
   - ``audit_trail``: Audit logging
   - ``compliance``: Regulatory compliance (FCRA, ECOA, GDPR)

**interfaces**
   API and UI configuration:
   
   - ``api``: REST API settings
   - ``cli``: CLI configuration
   - ``ui``: Web UI settings

Best Practices
--------------

Choosing the Right Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start Simple**
   
   Begin with beginner template, add features as needed:
   
   .. code-block:: python
   
      # Week 1: Beginner template
      config = load_template('beginner', 'basic_scorecard.yml')
      
      # Week 2: Add EDA and feature selection
      config = load_template('intermediate', 'full_scorecard.yml')
      
      # Production: Full monitoring
      config = load_template('advanced', 'production_scorecard.yml')

2. **Environment-Specific Configs**
   
   Use different templates for dev/staging/prod:
   
   .. code-block:: bash
   
      configs/
      ├── dev_config.yml        # Based on intermediate
      ├── staging_config.yml    # Based on advanced
      └── production_config.yml # Full advanced template

3. **Version Control**
   
   Keep all configs in git:
   
   .. code-block:: bash
   
      git add configs/*.yml
      git commit -m "Update scorecard configuration"

Customization Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

**Beginner Level**

Modify only:
- ``project.name``
- ``data`` paths
- ``scaling`` parameters

**Intermediate Level**

Additionally customize:
- ``features.selection`` method
- ``binning`` strategy
- ``model`` hyperparameters

**Advanced Level**

Full customization including:
- ``monitoring`` thresholds
- ``explainability`` settings
- ``governance`` requirements
- ``interfaces`` configuration

Template Inheritance
~~~~~~~~~~~~~~~~~~~~

Build custom templates from base templates:

.. code-block:: python

   from cr_score.templates import get_template_path
   import yaml
   
   # Load base template
   base_path = get_template_path('intermediate', 'full_scorecard.yml')
   with open(base_path) as f:
       base_config = yaml.safe_load(f)
   
   # Add company-specific defaults
   base_config['monitoring'] = {
       'performance': {'enabled': True, 'baseline_auc': 0.80},
       'drift': {'enabled': True, 'psi_threshold': 0.15}
   }
   
   # Save as company template
   with open('templates/company_standard.yml', 'w') as f:
       yaml.dump(base_config, f)

Production Monitoring Template
-------------------------------

Complete advanced template with monitoring:

.. code-block:: yaml

   # Advanced Production Template (excerpt)
   project:
     name: "production_scorecard"
     version: "2.0.0"
   
   monitoring:
     performance:
       enabled: true
       baseline_auc: 0.85
       alert_threshold_auc: 0.05
       storage_path: "./monitoring_data"
     
     drift:
       enabled: true
       psi_threshold: 0.1
       ks_threshold: 0.05
       check_frequency: "daily"
     
     predictions:
       enabled: true
       track_distributions: true
       detect_anomalies: true
     
     alerts:
       enabled: true
       channels: ["email", "slack"]
       severity_levels: ["warning", "critical"]
     
     metrics:
       enabled: true
       export_prometheus: true
       export_json: true
       collect_interval_seconds: 60
   
   explainability:
     shap:
       enabled: true
       sample_size: 1000
     reason_codes:
       enabled: true
       num_reasons: 4
     feature_importance:
       enabled: true
       methods: ["coefficient", "permutation"]
   
   governance:
     model_card:
       generate: true
       template: "default"
     audit_trail:
       enabled: true
       log_all_predictions: false
       retention_days: 90
     compliance:
       fcra: true
       ecoa: true
       gdpr: false

See Also
--------

- :doc:`/guides/configuration` - Configuration guide
- :doc:`/api/monitoring` - Monitoring and observability
- :doc:`/api/explainability` - Model explainability
- :doc:`/guides/quickstart` - Quick start guide

Template Files
--------------

All template files are located in:

.. code-block:: text

   src/cr_score/templates/
   ├── README.md
   ├── __init__.py
   ├── beginner/
   │   └── basic_scorecard.yml
   ├── intermediate/
   │   └── full_scorecard.yml
   └── advanced/
       └── production_scorecard.yml

See the `templates README <https://github.com/edmunlee87/CR_Score/tree/main/src/cr_score/templates/README.md>`_ for detailed documentation.
