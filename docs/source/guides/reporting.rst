Reporting Guide
===============

Generate professional HTML reports for your scorecards.

Why Generate Reports?
----------------------

Reports help you:

- **Document Your Work**: Complete record of model development
- **Share Results**: Easy to distribute to stakeholders
- **Present Findings**: Professional presentation format
- **Archive Models**: Save snapshots for compliance
- **Compare Models**: Side-by-side comparison of versions

What's in a Report?
-------------------

CR_Score generates comprehensive HTML reports with:

✅ **Executive Summary**

- Key performance metrics (AUC, Gini, KS)
- Number of features selected
- Feature selection method (if used)

✅ **Model Performance**

- Performance metrics table
- Interpretation of each metric
- Confusion matrix details

✅ **Interactive Visualizations**

- ROC curve
- Score distribution
- Score band analysis
- Calibration curve
- KS statistic curve

✅ **Feature Information**

- Selected features list
- IV (Information Value) for each feature
- Model coefficients

✅ **Model Configuration**

- PDO parameters
- Base score and base odds
- Calibration settings

Quick Start
-----------

Basic Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.reporting import HTMLReportGenerator
   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Build scorecard
   df = pd.read_csv("applications.csv")
   train, test = train_test_split(df, test_size=0.3, random_state=42)

   pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
   pipeline.fit(train, target_col="default")

   # Generate report
   generator = HTMLReportGenerator()
   report_path = generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=test.drop(columns=["default"]),
       y_test=test["default"],
       output_path="scorecard_report.html",
       title="Credit Scorecard Report",
       author="Risk Analytics Team"
   )

   print(f"Report saved to: {report_path}")

That's it! You now have a professional HTML report.

Opening the Report
~~~~~~~~~~~~~~~~~~

The report is saved as HTML and can be opened in any web browser:

.. code-block:: python

   import webbrowser

   # Open in default browser
   webbrowser.open(report_path)

Or double-click the HTML file in your file explorer.

Report Contents in Detail
-------------------------

1. Executive Summary
~~~~~~~~~~~~~~~~~~~~

At the top of every report, you'll find:

**Key Metrics Cards:**

- Number of features selected
- AUC (Area Under ROC Curve)
- Gini coefficient
- KS (Kolmogorov-Smirnov) statistic

**Feature Selection Info:**

If you used feature selection, the report shows:

- Selection method used
- Number of features before selection
- Number of features after selection

**Example Display:**

.. code-block:: text

   Features Selected: 8
   AUC Score: 0.825
   Gini Coefficient: 0.650
   KS Statistic: 0.447

   Feature Selection: Used stepwise method.
   Reduced from 25 to 8 features.

2. Model Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed table with:

**Metrics:**

- **AUC**: Discrimination ability (0.5-1.0, higher is better)
- **Gini**: 2 * AUC - 1, measures separation
- **KS**: Maximum difference between cumulative distributions
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted defaults, how many are correct
- **Recall**: Of actual defaults, how many are caught
- **F1 Score**: Harmonic mean of precision and recall

**Interpretation:**

Each metric includes an interpretation:

- "Excellent" for AUC ≥ 0.8
- "Good" for AUC ≥ 0.7
- "Fair" for AUC ≥ 0.6

3. Interactive Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All visualizations are **interactive**:

- Hover to see details
- Zoom and pan
- Export as PNG
- Full-screen mode

**Charts included:**

a. **ROC Curve**
   - Shows true positive rate vs false positive rate
   - AUC displayed in legend
   - Diagonal reference line

b. **Score Distribution**
   - Histogram of scores
   - Separate colors for good (green) and bad (red)
   - Shows separation quality

c. **Score Band Analysis**
   - Volume per score range (bars)
   - Default rate per range (line)
   - Validates score interpretation

d. **Calibration Curve**
   - Predicted probability vs actual rate
   - Perfect calibration line for reference
   - Shows if model is well-calibrated

e. **KS Statistic Curve**
   - Cumulative distributions
   - Maximum separation marked
   - Visual representation of KS statistic

4. Selected Features
~~~~~~~~~~~~~~~~~~~~

Complete list of features used in the scorecard:

- Feature name
- Information Value (IV)
- Model coefficient
- Contribution to the score

5. Model Configuration
~~~~~~~~~~~~~~~~~~~~~~

All configuration parameters:

- **PDO**: Points to Double Odds (e.g., 20)
- **Base Score**: Score at base odds (e.g., 600)
- **Base Odds**: Odds at base score (e.g., 50:1 = 2% default rate)
- **Calibration**: Whether intercept was adjusted
- **Feature Selection**: Method used (if any)

Step-by-Step Tutorial
----------------------

Step 1: Prepare Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Load data
   df = pd.read_csv("applications.csv")

   # Split
   train, test = train_test_split(
       df, test_size=0.3, random_state=42, stratify=df["default"]
   )

   print(f"Training: {len(train)} samples")
   print(f"Testing: {len(test)} samples")

Step 2: Build Your Scorecard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score import ScorecardPipeline

   pipeline = ScorecardPipeline(
       max_n_bins=5,
       min_iv=0.02,
       feature_selection="stepwise",
       max_features=10,
       pdo=20,
       base_score=600,
       base_odds=50.0,
       calibrate=True,
       random_state=42
   )

   pipeline.fit(train, target_col="default")

Step 3: Generate Report
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.reporting import HTMLReportGenerator

   generator = HTMLReportGenerator()

   report_path = generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=test.drop(columns=["default"]),
       y_test=test["default"],
       output_path="reports/scorecard_v1.html",
       title="Retail Credit Scorecard - v1.0",
       author="Credit Risk Team"
   )

   print(f"✓ Report generated: {report_path}")

Step 4: Open and Review
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import webbrowser

   # Open in browser
   webbrowser.open(report_path)

Step 5: Share with Team
~~~~~~~~~~~~~~~~~~~~~~~~

The HTML report is self-contained - just share the single file!

.. code-block:: python

   # Email it
   import smtplib
   from email.mime.text import MIMEText
   from email.mime.multipart import MIMEMultipart

   # (Email sending code here)

   # Or upload to shared drive
   import shutil
   shutil.copy(report_path, "/shared/reports/scorecard_v1.html")

Customization Options
---------------------

Custom Title and Author
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   report_path = generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=X_test,
       y_test=y_test,
       output_path="report.html",
       title="Personal Loan Scorecard - Production Model",
       author="John Smith, Senior Risk Analyst"
   )

Custom Output Location
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Organize by date
   from datetime import datetime

   date_str = datetime.now().strftime("%Y%m%d")
   output_path = f"reports/scorecard_{date_str}.html"

   report_path = generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=X_test,
       y_test=y_test,
       output_path=output_path
   )

Advanced Usage
--------------

Multiple Model Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate reports for multiple models and compare:

.. code-block:: python

   models = {
       "Baseline": pipeline_v1,
       "With Feature Selection": pipeline_v2,
       "XGBoost": pipeline_v3
   }

   for name, pipeline in models.items():
       output_path = f"reports/scorecard_{name.replace(' ', '_')}.html"

       generator.generate_scorecard_report(
           pipeline=pipeline,
           X_test=X_test,
           y_test=y_test,
           output_path=output_path,
           title=f"Credit Scorecard - {name}",
           author="Risk Analytics Team"
       )

   print("All reports generated! Compare them side-by-side.")

Batch Report Generation
~~~~~~~~~~~~~~~~~~~~~~~

Generate reports for different data segments:

.. code-block:: python

   segments = {
       "All Customers": test,
       "High Income": test[test["income"] > 75000],
       "Low Income": test[test["income"] <= 75000],
       "Young": test[test["age"] < 30],
       "Mature": test[test["age"] >= 30]
   }

   for segment_name, segment_data in segments.items():
       output_path = f"reports/scorecard_{segment_name.replace(' ', '_')}.html"

       generator.generate_scorecard_report(
           pipeline=pipeline,
           X_test=segment_data.drop(columns=["default"]),
           y_test=segment_data["default"],
           output_path=output_path,
           title=f"Credit Scorecard - {segment_name}",
           author="Risk Analytics Team"
       )

Best Practices
--------------

1. Use Descriptive Titles
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bad
   title="Report"

   # Good
   title="Retail Credit Scorecard - Validation Set - v2.1"

2. Include Version Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datetime import datetime

   version = "2.1"
   date = datetime.now().strftime("%Y-%m-%d")

   title = f"Credit Scorecard v{version} - Generated {date}"

3. Archive Reports
~~~~~~~~~~~~~~~~~~

Keep a history of all model versions:

.. code-block:: python

   import os
   from datetime import datetime

   # Create archive folder
   archive_dir = "reports/archive"
   os.makedirs(archive_dir, exist_ok=True)

   # Save with timestamp
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_path = f"{archive_dir}/scorecard_{timestamp}.html"

   generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=X_test,
       y_test=y_test,
       output_path=output_path
   )

4. Add Context in Title
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Include important context
   title = (
       "Personal Loan Scorecard - "
       "Validation Period: Jan-Mar 2024 - "
       "Sample: 15,000 applications"
   )

5. Generate After Every Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # In your experiment loop
   for config in configurations:
       pipeline = ScorecardPipeline(**config)
       pipeline.fit(train, target_col="default")

       # Generate report immediately
       generator.generate_scorecard_report(
           pipeline=pipeline,
           X_test=X_test,
           y_test=y_test,
           output_path=f"reports/experiment_{config['name']}.html",
           title=f"Experiment: {config['name']}"
       )

Report Structure (Technical)
-----------------------------

For those interested in the HTML structure:

**Technologies Used:**

- HTML5 for structure
- CSS3 for styling (embedded)
- Plotly.js for interactive charts (embedded)
- Self-contained (no external dependencies)

**File Size:**

- Typical report: 2-5 MB
- Includes all JavaScript and CSS
- Includes all chart data
- Works offline

**Compatibility:**

- ✅ All modern browsers (Chrome, Firefox, Safari, Edge)
- ✅ Mobile-friendly (responsive design)
- ✅ Print-friendly
- ✅ Screen reader accessible

Common Issues and Solutions
----------------------------

Issue 1: Report Too Large
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** HTML file is very large (> 10 MB)

**Solution:** Reduce number of data points in visualizations

.. code-block:: python

   # Sample test data if too large
   if len(test) > 10000:
       test_sample = test.sample(n=10000, random_state=42)
   else:
       test_sample = test

   generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=test_sample.drop(columns=["default"]),
       y_test=test_sample["default"],
       output_path="report.html"
   )

Issue 2: Charts Not Interactive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Charts don't respond to mouse

**Solution:** Make sure JavaScript is enabled in your browser

Issue 3: Report Doesn't Open
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Double-clicking HTML doesn't work

**Solution:** Right-click → Open With → Choose browser

Or use Python:

.. code-block:: python

   import webbrowser
   webbrowser.open(report_path)

Integration with Workflow
--------------------------

With MLflow Tracking
~~~~~~~~~~~~~~~~~~~~

Log reports as artifacts in MLflow:

.. code-block:: python

   import mlflow

   with mlflow.start_run():
       # Train model
       pipeline.fit(train, target_col="default")

       # Log metrics
       metrics = pipeline.evaluate(test, target_col="default")
       mlflow.log_metrics(metrics)

       # Generate and log report
       report_path = generator.generate_scorecard_report(
           pipeline=pipeline,
           X_test=X_test,
           y_test=y_test,
           output_path="temp_report.html"
       )

       mlflow.log_artifact(report_path, "reports")

With CI/CD Pipeline
~~~~~~~~~~~~~~~~~~~

Generate reports automatically in CI/CD:

.. code-block:: python

   # In your CI/CD script
   def generate_validation_report():
       # Load latest model
       pipeline = load_model("models/latest.pkl")

       # Load validation data
       test = load_validation_data()

       # Generate report
       generator = HTMLReportGenerator()
       report_path = generator.generate_scorecard_report(
           pipeline=pipeline,
           X_test=test.drop(columns=["default"]),
           y_test=test["default"],
           output_path="reports/validation_latest.html",
           title="Automated Validation Report"
       )

       # Upload to S3/shared location
       upload_to_s3(report_path, "bucket/validation-reports/")

Complete Example
----------------

Here's a complete end-to-end example:

.. code-block:: python

   from cr_score import ScorecardPipeline
   from cr_score.reporting import HTMLReportGenerator
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from datetime import datetime
   import webbrowser
   import os

   # Configuration
   OUTPUT_DIR = "reports"
   MODEL_VERSION = "1.0"

   # Create output directory
   os.makedirs(OUTPUT_DIR, exist_ok=True)

   # Load data
   print("Loading data...")
   df = pd.read_csv("applications.csv")
   train, test = train_test_split(df, test_size=0.3, random_state=42)

   # Build scorecard
   print("Building scorecard...")
   pipeline = ScorecardPipeline(
       max_n_bins=5,
       feature_selection="stepwise",
       max_features=10,
       pdo=20,
       base_score=600,
       random_state=42
   )

   pipeline.fit(train, target_col="default")

   # Evaluate
   metrics = pipeline.evaluate(test, target_col="default")
   print(f"AUC: {metrics['auc']:.3f}")

   # Generate report
   print("Generating report...")
   generator = HTMLReportGenerator()

   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_path = f"{OUTPUT_DIR}/scorecard_v{MODEL_VERSION}_{timestamp}.html"

   report_path = generator.generate_scorecard_report(
       pipeline=pipeline,
       X_test=test.drop(columns=["default"]),
       y_test=test["default"],
       output_path=output_path,
       title=f"Credit Scorecard v{MODEL_VERSION} - Validation Report",
       author="Risk Analytics Team"
   )

   print(f"✓ Report saved to: {report_path}")

   # Open in browser
   print("Opening report in browser...")
   webbrowser.open(report_path)

   print("Done!")

Next Steps
----------

- :doc:`/api/reporting` - Complete API reference
- :doc:`/guides/visualization` - Create custom visualizations
- :doc:`/examples/complete_workflow` - See reporting in action
- :doc:`/api/pipeline` - Build scorecards to report on
