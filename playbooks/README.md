# CR_Score Playbooks

Interactive Jupyter notebook tutorials for learning CR_Score, from basic to advanced.

## Overview

These playbooks provide hands-on, runnable examples that teach you how to build credit scorecards using CR_Score. Each notebook is self-contained and progressively introduces new concepts.

## Prerequisites

### Required
- Python 3.9+
- CR_Score installed: `pip install -e .` (from project root)
- Jupyter: `pip install jupyter notebook`

### Optional  
- PySpark 3.4+ (only for Playbook 05 - Advanced Topics)
- MLflow (for experiment tracking in Playbook 02)

**Don't have PySpark?** No problem! Playbooks 01-04 work perfectly without it. The `spark_helper.py` module gracefully handles missing PySpark installations.

## Playbook Structure

### 01: Quick Start (Beginner, 5-10 min)
**File:** `01_quickstart.ipynb`

Build your first credit scorecard in 3 lines of code!

**What you'll learn:**
- Load credit application data
- Build a complete scorecard with ScorecardPipeline
- Evaluate performance (AUC, Gini, KS)
- Score new applications
- Make approve/decline decisions

**No PySpark required**

---

### 02: Feature Selection (Intermediate, 15-20 min)
**File:** `02_feature_selection.ipynb`

Master model-agnostic feature selection methods.

**What you'll learn:**
- Forward selection (greedy addition)
- Backward elimination (greedy removal)
- Stepwise selection (bidirectional)
- Exhaustive search (all combinations)
- MLflow integration for tracking experiments
- Compare feature selection results

**No PySpark required**

---

### 03: Visualization & Reporting (Intermediate, 15-20 min)
**File:** `03_visualization_reporting.ipynb`

Create beautiful, interactive visualizations and professional HTML reports.

**What you'll learn:**
- Binning visualizations (WoE, IV, distributions)
- Scorecard performance plots (ROC, KS, calibration)
- Score distribution analysis
- Generate comprehensive HTML reports
- Export for stakeholder presentations

**No PySpark required**

---

### 04: Complete Workflow (Intermediate, 25-30 min)
**File:** `04_complete_workflow.ipynb`

Master the end-to-end scorecard development process.

**What you'll learn:**
- Data loading and validation
- Exploratory data analysis (EDA)
- Optimal binning with OptBinning
- WoE encoding
- Feature selection
- Model training with sample weights
- Calibration and PDO scaling
- Full scorecard deployment

**No PySpark required**

---

### 05: Advanced Topics (Advanced, 30-40 min)
**File:** `05_advanced_topics.ipynb`

Explore enterprise-grade features for production deployment.

**What you'll learn:**
- Config-driven scorecard development (YAML)
- Spark-based data compression
- Reject inference (parceling, reweighting)
- Drift detection (PSI, CSI)
- MCP tools for AI agent integration
- Artifact management and versioning
- Production deployment patterns

**PySpark recommended for this playbook**

---

## Getting Started

### 1. Setup Environment

```bash
# Navigate to playbooks folder
cd playbooks

# Install Jupyter (if not already installed)
pip install jupyter notebook matplotlib

# Generate sample data (already done if you see data/ folder)
python data/generate_sample_data.py
```

### 2. Launch Jupyter

```bash
jupyter notebook
```

This will open Jupyter in your browser.

### 3. Start with Playbook 01

Open `01_quickstart.ipynb` and run through it cell by cell (Shift + Enter).

### 4. Progress Through Playbooks

Complete the playbooks in order:
1. 01_quickstart.ipynb (Start here!)
2. 02_feature_selection.ipynb
3. 03_visualization_reporting.ipynb
4. 04_complete_workflow.ipynb
5. 05_advanced_topics.ipynb (Requires PySpark)

## Sample Data

The `data/` folder contains synthetic credit application data:

- **credit_applications.csv**: Full dataset (5,000 applications)
- **train.csv**: Training set (3,500 applications, 70%)
- **test.csv**: Test set (1,500 applications, 30%)

**Features:**
- Demographics: age, education, employment_type, home_ownership
- Financial: income, debt_to_income_ratio, loan_amount
- Credit: credit_history_years, num_credit_lines, credit_utilization, num_delinquent_accounts
- Target: default (1 = defaulted, 0 = no default)

**Data Characteristics:**
- Realistic distributions
- ~0.5% default rate (low default scenario)
- Multiple numeric and categorical features
- No missing values (for simplicity)

## PySpark Handling

Don't have PySpark installed? **No problem!**

The `spark_helper.py` module provides:
- Automatic detection of PySpark availability
- Graceful fallback to pandas mode
- Clear messaging about what's available
- All tutorials work without PySpark (except advanced Spark compression)

When you run the notebooks, you'll see:

**With PySpark:**
```
[OK] PySpark is available
```

**Without PySpark:**
```
[INFO] PySpark not available - using pandas mode (this is fine for tutorials!)

[NOTE] PySpark is not installed

This is perfectly fine! All CR_Score tutorials work without PySpark for small/medium datasets.

[OK] You can use: pandas mode (default)
[X] You cannot use: Spark compression (only needed for very large datasets)

To install PySpark (optional):
    pip install pyspark>=3.4.0
```

## Tips for Success

### Running Notebooks
- Run cells in order (Shift + Enter)
- Wait for each cell to complete before running the next
- Look for `[OK]` messages indicating success
- Read the markdown explanations between code cells

### Troubleshooting
- **Import Error**: Make sure CR_Score is installed (`pip install -e .` from project root)
- **File Not Found**: Run notebooks from the `playbooks/` directory
- **Missing Data**: Run `python data/generate_sample_data.py`
- **PySpark Warnings**: Ignore if you don't have PySpark (it's optional!)

### Best Practices
- Start with Playbook 01 even if you're experienced
- Run all cells in a notebook to see the complete workflow
- Experiment! Modify parameters and see what changes
- Save your modified notebooks with different names to preserve originals

## Learning Path

### For Beginners
1. Start with **Playbook 01** (Quick Start)
2. Understand the 3-line scorecard approach
3. Move to **Playbook 03** (Visualization) to see your results
4. Then try **Playbook 02** (Feature Selection) for model improvement

### For Intermediate Users
1. Quick review of **Playbook 01**
2. Deep dive into **Playbook 04** (Complete Workflow)
3. Experiment with **Playbook 02** (Feature Selection)
4. Master **Playbook 03** (Visualization)

### For Advanced Users
1. Skim **Playbooks 01-04** for syntax
2. Focus on **Playbook 05** (Advanced Topics)
3. Explore config-driven development
4. Implement production patterns

## Next Steps After Playbooks

Once you've completed the playbooks:

1. **Build Your Own Scorecard**
   - Use your own data
   - Apply the patterns from the playbooks
   - Customize binning and scoring parameters

2. **Explore the Documentation**
   - API Reference: `docs/build/html/api/index.html`
   - User Guides: `docs/build/html/guides/index.html`
   - Examples: `examples/` folder

3. **Production Deployment**
   - Review **Playbook 05** for production patterns
   - Implement config-driven workflows
   - Set up artifact versioning
   - Enable audit logging

4. **Contribute**
   - Share your own playbooks
   - Report issues on GitHub
   - Suggest improvements

## Support

### Questions?
- Check the main README: `../README.md`
- Read the docs: `../docs/build/html/index.html`
- See examples: `../examples/`

### Issues?
- Open an issue on GitHub
- Include the playbook name and error message
- Share the cell that's failing

## File Structure

```
playbooks/
├── README.md                          # This file
├── spark_helper.py                    # PySpark graceful handling
├── generate_notebooks.py              # Notebook generator script
├── data/
│   ├── generate_sample_data.py        # Data generation script
│   ├── credit_applications.csv        # Full dataset
│   ├── train.csv                      # Training set
│   └── test.csv                       # Test set
├── 01_quickstart.ipynb                # Beginner: 3-line scorecard
├── 02_feature_selection.ipynb         # Intermediate: Feature selection
├── 03_visualization_reporting.ipynb   # Intermediate: Viz & reports
├── 04_complete_workflow.ipynb         # Intermediate: Full workflow
└── 05_advanced_topics.ipynb           # Advanced: Production features
```

## License

Same as CR_Score project.

---

**Ready to start?** Launch Jupyter and open `01_quickstart.ipynb`!

```bash
jupyter notebook 01_quickstart.ipynb
```

Happy learning!
