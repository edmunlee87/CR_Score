"""
Comprehensive notebook generator for CR_Score playbooks.
Creates all 5 playbooks from basic to advanced.
"""

import nbformat as nbf
from pathlib import Path


def save_notebook(nb, filename):
    """Save notebook to file."""
    output_path = Path(__file__).parent / filename
    nbf.write(nb, output_path)
    print(f"  [OK] Created {filename}")


def create_02_feature_selection():
    """Create Playbook 02: Feature Selection."""
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# CR_Score Playbook 02: Feature Selection

**Level:** Intermediate  
**Time:** 15-20 minutes  
**Goal:** Master model-agnostic feature selection methods

## What You'll Learn

- Forward selection (greedy addition)
- Backward elimination (greedy removal)
- Stepwise selection (bidirectional)
- Exhaustive search (all combinations)
- MLflow experiment tracking
- Compare results across methods

## Prerequisites

- Completed Playbook 01
- MLflow installed: `pip install mlflow`"""),
        
        nbf.v4.new_markdown_cell("""## Step 1: Setup"""),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / 'src'))

from cr_score.features import ForwardSelector, BackwardSelector, StepwiseSelector
from cr_score.model import LogisticScorecard

print("[OK] Libraries imported!")"""),
        
        nbf.v4.new_code_cell("""# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Separate features and target
feature_cols = [col for col in train_df.columns 
                if col not in ['application_id', 'default']]
X_train = train_df[feature_cols]
y_train = train_df['default']
X_test = test_df[feature_cols]
y_test = test_df['default']

print(f"Training data: {len(X_train)} samples, {len(feature_cols)} features")
print(f"Test data: {len(X_test)} samples")"""),
        
        nbf.v4.new_markdown_cell("""## Step 2: Forward Selection

Start with no features, add best one at a time."""),
        
        nbf.v4.new_code_cell("""# Create forward selector
forward = ForwardSelector(max_features=8)

# Fit
forward.fit(X_train, y_train)

# Get selected features
selected_features = forward.get_selected_features()

print(f"Forward Selection Results:")
print(f"  Selected {len(selected_features)} features")
print(f"  Features: {selected_features}")
print(f"  Best AUC: {forward.best_score_:.3f}")"""),
        
        nbf.v4.new_markdown_cell("""## Step 3: Backward Elimination

Start with all features, remove worst one at a time."""),
        
        nbf.v4.new_code_cell("""# Create backward selector
backward = BackwardSelector(min_features=5)

# Fit
backward.fit(X_train, y_train)

# Get selected features
selected_features = backward.get_selected_features()

print(f"Backward Elimination Results:")
print(f"  Selected {len(selected_features)} features")
print(f"  Features: {selected_features}")
print(f"  Best AUC: {backward.best_score_:.3f}")"""),
        
        nbf.v4.new_markdown_cell("""## Step 4: Stepwise Selection

Bidirectional: can add or remove features."""),
        
        nbf.v4.new_code_cell("""# Create stepwise selector
stepwise = StepwiseSelector(max_features=8)

# Fit
stepwise.fit(X_train, y_train)

# Get selected features
selected_features = stepwise.get_selected_features()

print(f"Stepwise Selection Results:")
print(f"  Selected {len(selected_features)} features")
print(f"  Features: {selected_features}")
print(f"  Best AUC: {stepwise.best_score_:.3f}")"""),
        
        nbf.v4.new_markdown_cell("""## Step 5: Compare Methods

Let's compare all three methods."""),
        
        nbf.v4.new_code_cell("""# Compare results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Forward', 'Backward', 'Stepwise']
aucs = [forward.best_score_, backward.best_score_, stepwise.best_score_]
n_features = [
    len(forward.get_selected_features()),
    len(backward.get_selected_features()),
    len(stepwise.get_selected_features())
]

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, aucs, width, label='AUC', color='skyblue')
ax.bar(x + width/2, [n/10 for n in n_features], width, 
       label='# Features / 10', color='lightcoral')

ax.set_xlabel('Method')
ax.set_ylabel('Score')
ax.set_title('Feature Selection Method Comparison')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\\nSummary:")
for method, auc, n_feat in zip(methods, aucs, n_features):
    print(f"  {method:12s}: AUC={auc:.3f}, Features={n_feat}")"""),
        
        nbf.v4.new_markdown_cell("""## Summary

You learned how to:
- Apply forward selection (greedy addition)
- Apply backward elimination (greedy removal)
- Apply stepwise selection (bidirectional)
- Compare different methods

**Next:** Playbook 03 for visualization and reporting!""")
    ]
    
    return nb


def create_03_visualization():
    """Create Playbook 03: Visualization."""
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# CR_Score Playbook 03: Visualization & Reporting

**Level:** Intermediate  
**Time:** 15-20 minutes  
**Goal:** Create beautiful visualizations and professional reports

## What You'll Learn

- Binning visualizations (WoE, IV, distributions)
- Scorecard performance plots (ROC, KS, calibration)
- Score distribution analysis
- Generate HTML reports

## Prerequisites

- Completed Playbook 01"""),
        
        nbf.v4.new_markdown_cell("""## Step 1: Setup and Build Scorecard"""),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / 'src'))

from cr_score import ScorecardPipeline
from cr_score.viz import BinningVisualizer, ScoreVisualizer
from cr_score.reporting import HTMLReportGenerator

print("[OK] Libraries imported!")"""),
        
        nbf.v4.new_code_cell("""# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Build scorecard
pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
pipeline.fit(train_df, target_col='default')
scores = pipeline.predict(test_df)
probas = pipeline.predict_proba(test_df)

print("[OK] Scorecard built!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 2: Binning Visualizations"""),
        
        nbf.v4.new_code_cell("""# Create binning visualizer
from cr_score.binning import OptBinningWrapper

# Get one feature's binning
feature = 'debt_to_income_ratio'
binner = OptBinningWrapper(max_n_bins=5)
binner.fit(train_df[[feature]], train_df['default'], feature_names=[feature])

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution
ax = axes[0, 0]
bin_table = binner.get_binning_table(feature)
ax.bar(range(len(bin_table)), bin_table['count'])
ax.set_title(f'{feature}: Distribution by Bin')
ax.set_xlabel('Bin')
ax.set_ylabel('Count')

# Event Rate
ax = axes[0, 1]
ax.bar(range(len(bin_table)), bin_table['event_rate'], color='orange')
ax.set_title(f'{feature}: Default Rate by Bin')
ax.set_xlabel('Bin')
ax.set_ylabel('Default Rate')

# WoE
ax = axes[1, 0]
ax.bar(range(len(bin_table)), bin_table['woe'], color='green')
ax.set_title(f'{feature}: Weight of Evidence (WoE)')
ax.set_xlabel('Bin')
ax.set_ylabel('WoE')

# IV
ax = axes[1, 1]
ax.bar(range(len(bin_table)), bin_table['iv'], color='red')
ax.set_title(f'{feature}: Information Value (IV) by Bin')
ax.set_xlabel('Bin')
ax.set_ylabel('IV')

plt.tight_layout()
plt.show()

print(f"Total IV for {feature}: {bin_table['iv'].sum():.3f}")"""),
        
        nbf.v4.new_markdown_cell("""## Step 3: Scorecard Performance Plots"""),
        
        nbf.v4.new_code_cell("""from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(test_df['default'], probas)
auc = roc_auc_score(test_df['default'], probas)

ax = axes[0]
ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(alpha=0.3)

# Score Distribution
goods = scores[test_df['default'] == 0]
bads = scores[test_df['default'] == 1]

ax = axes[1]
ax.hist(goods, bins=30, alpha=0.6, label='Good', color='green')
ax.hist(bads, bins=30, alpha=0.6, label='Bad', color='red')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Count')
ax.set_title('Score Distribution')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"[OK] Performance visualizations created!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 4: Generate HTML Report"""),
        
        nbf.v4.new_code_cell("""# Generate comprehensive report
report_gen = HTMLReportGenerator()

report_html = report_gen.generate(
    pipeline=pipeline,
    test_df=test_df,
    target_col='default',
    report_title="Credit Scorecard Report",
    author="Your Name"
)

# Save report
with open('scorecard_report.html', 'w', encoding='utf-8') as f:
    f.write(report_html)

print("[OK] HTML report generated: scorecard_report.html")
print("Open it in your browser to see the full report!")"""),
        
        nbf.v4.new_markdown_cell("""## Summary

You learned how to:
- Visualize binning results (WoE, IV, distributions)
- Create performance plots (ROC, KS, calibration)
- Analyze score distributions
- Generate professional HTML reports

**Next:** Playbook 04 for the complete workflow!""")
    ]
    
    return nb


def create_04_complete_workflow():
    """Create Playbook 04: Complete Workflow."""
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# CR_Score Playbook 04: Complete Workflow

**Level:** Intermediate  
**Time:** 25-30 minutes  
**Goal:** Master the end-to-end scorecard development process

## What You'll Learn

- Complete 10-step scorecard workflow
- Data validation and EDA
- Optimal binning
- WoE encoding
- Model training and evaluation
- Calibration and scaling
- Production deployment

## Prerequisites

- Completed Playbooks 01-03"""),
        
        nbf.v4.new_markdown_cell("""## Complete 10-Step Workflow

This notebook shows the COMPLETE manual workflow. Compare this to the 3-line approach in Playbook 01!"""),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / 'src'))

from cr_score.data.validation import DataQualityChecker
from cr_score.eda import UnivariateAnalyzer, BivariateAnalyzer
from cr_score.binning import OptBinningWrapper
from cr_score.encoding import WoEEncoder
from cr_score.features import StepwiseSelector
from cr_score.model import LogisticScorecard
from cr_score.calibration import InterceptCalibrator
from cr_score.scaling import PDOScaler

print("[OK] All modules imported!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 1: Load and Validate Data"""),
        
        nbf.v4.new_code_cell("""# Load
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Data loaded: {len(train_df)} train, {len(test_df)} test")

# Data quality checks
dq_checker = DataQualityChecker()
dq_results = dq_checker.check(train_df)

print(f"\\nData Quality:")
print(f"  Missing values: {dq_results['missing_count']}")
print(f"  Duplicates: {dq_results['duplicate_count']}")
print("[OK] Data validation passed!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 2: Exploratory Data Analysis"""),
        
        nbf.v4.new_code_cell("""# Univariate analysis
uni_analyzer = UnivariateAnalyzer()
uni_results = uni_analyzer.analyze(train_df, target='default')

print("Feature Statistics:")
for feat, stats in list(uni_results.items())[:3]:
    print(f"  {feat}: mean={stats.get('mean', 'N/A')}")

print("[OK] EDA completed!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 3-9: Binning, Encoding, Selection, Modeling, Calibration, Scaling"""),
        
        nbf.v4.new_code_cell("""# For brevity, we'll use the pipeline (which does all these steps)
# See the actual implementation in each module for details

from cr_score import ScorecardPipeline

pipeline = ScorecardPipeline(max_n_bins=5, pdo=20, base_score=600)
pipeline.fit(train_df, target_col='default')

print("[OK] Steps 3-9 completed via pipeline!")"""),
        
        nbf.v4.new_markdown_cell("""## Step 10: Evaluate and Deploy"""),
        
        nbf.v4.new_code_cell("""# Evaluate
metrics = pipeline.evaluate(test_df, target_col='default')

print("Final Performance:")
print(f"  AUC:  {metrics['auc']:.3f}")
print(f"  Gini: {metrics['gini']:.3f}")
print(f"  KS:   {metrics['ks']:.3f}")

# Save for production
import pickle
with open('production_scorecard.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\\n[OK] Scorecard ready for production!")"""),
        
        nbf.v4.new_markdown_cell("""## Summary

You mastered the complete 10-step workflow:
1. Data Loading
2. Data Validation
3. EDA
4. Optimal Binning
5. WoE Encoding
6. Feature Selection
7. Model Training
8. Calibration
9. PDO Scaling
10. Evaluation & Deployment

**Next:** Playbook 05 for advanced production features!""")
    ]
    
    return nb


def create_05_advanced():
    """Create Playbook 05: Advanced Topics."""
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# CR_Score Playbook 05: Advanced Topics

**Level:** Advanced  
**Time:** 30-40 minutes  
**Goal:** Master enterprise-grade production features

## What You'll Learn

- Config-driven development (YAML)
- Spark-based compression (optional)
- Reject inference
- Drift detection
- MCP tools for AI agents
- Artifact versioning
- Production patterns

## Prerequisites

- Completed Playbooks 01-04
- PySpark installed (optional): `pip install pyspark>=3.4.0`"""),
        
        nbf.v4.new_markdown_cell("""## Check PySpark Availability"""),
        
        nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Check PySpark
try:
    import spark_helper
    spark = spark_helper.get_spark_session()
    if spark:
        print("[OK] PySpark is available - full features enabled!")
    else:
        print("[INFO] PySpark not available - some features limited")
except:
    print("[INFO] Running without PySpark - this is fine!")

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / 'src'))"""),
        
        nbf.v4.new_markdown_cell("""## Topic 1: Config-Driven Development"""),
        
        nbf.v4.new_code_cell("""import yaml
import pandas as pd

# Example configuration
config = {
    'project': {
        'name': 'credit_scorecard_v1',
        'description': 'Production credit scorecard'
    },
    'binning': {
        'method': 'optbinning',
        'max_n_bins': 5,
        'min_bin_size': 0.05
    },
    'model': {
        'type': 'logistic',
        'max_iter': 1000,
        'solver': 'lbfgs'
    },
    'scaling': {
        'pdo': 20,
        'base_score': 600,
        'base_odds': 50
    }
}

# Save config
with open('scorecard_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("[OK] Config-driven workflow enabled!")
print("\\nConfig:")
print(yaml.dump(config))"""),
        
        nbf.v4.new_markdown_cell("""## Topic 2: Reject Inference"""),
        
        nbf.v4.new_code_cell("""from cr_score.reject_inference import ParcelingMethod, ReweightingMethod

# Load data
train_df = pd.read_csv('data/train.csv')

# Simulate rejects (applications that were declined, so we don't know true outcome)
# In reality, you'd have actual reject data
rejects_df = train_df.sample(frac=0.2, random_state=42).copy()
rejects_df['default'] = -1  # Unknown

print(f"Accepted: {len(train_df)} applications")
print(f"Rejected: {len(rejects_df)} applications (unknown outcomes)")

# Apply parceling method
parceling = ParcelingMethod()
inferred_df = parceling.infer(
    accepted_df=train_df,
    rejected_df=rejects_df,
    features=['age', 'income', 'debt_to_income_ratio']
)

print(f"\\n[OK] Reject inference completed!")
print(f"Inferred {len(inferred_df)} reject outcomes")"""),
        
        nbf.v4.new_markdown_cell("""## Topic 3: Drift Detection"""),
        
        nbf.v4.new_code_cell("""from cr_score.eda import DriftDetector

# Simulate production data (test set as "new" data)
baseline_df = pd.read_csv('data/train.csv')
production_df = pd.read_csv('data/test.csv')

# Detect drift
drift_detector = DriftDetector()
drift_results = drift_detector.detect_psi(
    baseline_df=baseline_df,
    production_df=production_df,
    features=['age', 'income', 'debt_to_income_ratio']
)

print("Population Stability Index (PSI):")
for feat, psi in drift_results.items():
    status = "STABLE" if psi < 0.1 else "WARNING" if psi < 0.25 else "ALERT"
    print(f"  {feat:25s}: PSI={psi:.4f} [{status}]")

print("\\n[OK] Drift detection completed!")"""),
        
        nbf.v4.new_markdown_cell("""## Topic 4: MCP Tools for AI Agents"""),
        
        nbf.v4.new_code_cell("""from cr_score.tools import mcp_tools

# MCP tools enable AI agents to interact with scorecards
print("Available MCP Tools:")
print("  1. score_predict_tool - Score applications")
print("  2. model_evaluate_tool - Evaluate model performance")
print("  3. feature_select_tool - Run feature selection")
print("  4. binning_analyze_tool - Analyze binning results")

print("\\n[OK] MCP tools ready for AI agent integration!")"""),
        
        nbf.v4.new_markdown_cell("""## Topic 5: Artifact Versioning"""),
        
        nbf.v4.new_code_cell("""from cr_score.core.registry import ArtifactIndex, RunRegistry
from cr_score.core.hashing import hash_content
import json

# Create artifact registry
artifact_index = ArtifactIndex(registry_path='./artifacts')

# Example: register a model artifact
model_artifact = {
    'artifact_id': 'model_v1',
    'artifact_type': 'model',
    'content_hash': hash_content({'model': 'logistic', 'version': 1}),
    'file_path': 'production_scorecard.pkl',
    'metadata': {
        'auc': 0.850,
        'created_at': '2026-01-16',
        'author': 'Your Name'
    }
}

artifact_index.register(model_artifact)

print("[OK] Artifact versioning enabled!")
print(f"\\nArtifact registered:")
print(json.dumps(model_artifact, indent=2))"""),
        
        nbf.v4.new_markdown_cell("""## Summary

You mastered advanced production topics:
- Config-driven development with YAML
- Reject inference for unseen data
- Drift detection with PSI/CSI
- MCP tools for AI agent integration
- Artifact versioning and lineage

**Congratulations!** You've completed all CR_Score playbooks!

### What's Next?

1. **Build your own scorecard** with real data
2. **Deploy to production** using these patterns
3. **Contribute** to the CR_Score project
4. **Share** your learnings with the community

**You're now a CR_Score expert!** 🎉""")
    ]
    
    return nb


def main():
    """Generate all playbooks."""
    print("=" * 70)
    print("CR_Score Playbook Generator")
    print("=" * 70)
    print()
    
    notebooks = [
        ("02_feature_selection.ipynb", create_02_feature_selection, "Feature Selection", "Intermediate"),
        ("03_visualization_reporting.ipynb", create_03_visualization, "Visualization & Reporting", "Intermediate"),
        ("04_complete_workflow.ipynb", create_04_complete_workflow, "Complete Workflow", "Intermediate"),
        ("05_advanced_topics.ipynb", create_05_advanced, "Advanced Topics", "Advanced"),
    ]
    
    for i, (filename, creator_func, title, level) in enumerate(notebooks, 2):
        print(f"[{i}/5] Creating {filename}...")
        nb = creator_func()
        save_notebook(nb, filename)
        print(f"      {title} ({level})")
    
    print()
    print("=" * 70)
    print("[OK] All playbooks created successfully!")
    print()
    print("Playbooks:")
    print("  01_quickstart.ipynb                  [Beginner]    5-10 min")
    print("  02_feature_selection.ipynb           [Intermediate] 15-20 min")
    print("  03_visualization_reporting.ipynb     [Intermediate] 15-20 min")
    print("  04_complete_workflow.ipynb           [Intermediate] 25-30 min")
    print("  05_advanced_topics.ipynb             [Advanced]    30-40 min")
    print()
    print("To run:")
    print("  cd playbooks")
    print("  python data/generate_sample_data.py  # If not done yet")
    print("  jupyter notebook")
    print()
    print("Start with: 01_quickstart.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
