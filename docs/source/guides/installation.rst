Installation Guide
==================

Requirements
------------

- Python 3.9, 3.10, or 3.11
- pip (Python package manager)
- git (for cloning repository)

Recommended (Optional):
- Apache Spark 3.4+ (for large datasets > 10M rows)
- Java 8 or 11 (for Spark)

Standard Installation
---------------------

Clone and Install
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/edmunlee87/CR_Score.git
   cd CR_Score

   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate      # Windows

   # Install package
   pip install -e ".[dev]"

This installs CR_Score in development mode with all dependencies.

Production Installation
-----------------------

For production deployment:

.. code-block:: bash

   pip install -e .

This installs only runtime dependencies (no development tools).

Verify Installation
-------------------

Run the verification script:

.. code-block:: bash

   python verify_installation.py

You should see:

.. code-block:: text

   ✓ Core - Config                       - OK
   ✓ Core - Logging                      - OK
   ✓ Data - Connectors                   - OK
   ...
   ✓ Pipeline                            - OK

   ✅ Verification PASSED - All 21 components working!

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- **pyspark** >= 3.4.0 - Big data processing
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical computing
- **scipy** >= 1.10.0 - Statistical functions
- **scikit-learn** >= 1.3.0 - Machine learning
- **optbinning** >= 0.19.0 - Optimal binning
- **mlflow** >= 2.10.0 - Experiment tracking

Configuration & Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **pydantic** >= 2.0.0 - Schema validation
- **pyyaml** >= 6.0 - YAML parsing
- **structlog** >= 23.1.0 - Structured logging

Visualization & Reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **plotly** >= 5.15.0 - Interactive charts
- **jinja2** >= 3.1.0 - HTML templating

CLI & API
~~~~~~~~~

- **click** >= 8.1.0 - CLI framework
- **fastapi** >= 0.100.0 - REST API (optional)
- **uvicorn** >= 0.23.0 - ASGI server (optional)

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

- **pytest** >= 7.0.0 - Testing
- **pytest-cov** >= 4.0.0 - Coverage
- **flake8** >= 6.0.0 - Linting
- **black** >= 23.0.0 - Formatting
- **mypy** >= 1.0.0 - Type checking
- **sphinx** >= 7.0.0 - Documentation
- **sphinx-rtd-theme** >= 1.3.0 - Documentation theme

Troubleshooting
---------------

ImportError for optbinning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you see ``ImportError: No module named 'optbinning'``:

.. code-block:: bash

   pip install optbinning>=0.19.0

PySpark Issues
~~~~~~~~~~~~~~

If PySpark fails to start:

1. Check Java installation:

.. code-block:: bash

   java -version

2. Install Java 8 or 11 if missing
3. Set ``JAVA_HOME`` environment variable

MLflow Tracking Issues
~~~~~~~~~~~~~~~~~~~~~~

If MLflow UI doesn't start:

.. code-block:: bash

   pip install --upgrade mlflow
   mlflow ui

Windows-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter path issues on Windows:

1. Use PowerShell (not CMD)
2. Activate venv: ``.venv\Scripts\activate``
3. Use backslashes in paths or raw strings

Docker Installation
-------------------

For Docker deployment:

.. code-block:: dockerfile

   FROM python:3.10-slim

   WORKDIR /app

   # Install Java for Spark
   RUN apt-get update && \
       apt-get install -y openjdk-11-jre-headless && \
       rm -rf /var/lib/apt/lists/*

   # Copy and install
   COPY . .
   RUN pip install -e .

   # Run
   CMD ["python", "-m", "cr_score"]

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   cd CR_Score
   git pull origin main
   pip install -e ".[dev]" --upgrade

Uninstallation
--------------

To uninstall:

.. code-block:: bash

   pip uninstall cr-score

Next Steps
----------

- :doc:`/guides/quickstart` - Build your first scorecard
- :doc:`/guides/configuration` - Configure CR_Score
- :doc:`/api/pipeline` - API Reference
