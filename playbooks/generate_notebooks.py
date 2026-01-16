"""
Generate CR_Score playbook notebooks.

This script creates comprehensive, runnable Jupyter notebooks from basic to advanced.
"""

import nbformat as nbf
from pathlib import Path


def create_01_quickstart():
    """Create Playbook 01: Quickstart notebook."""
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title and intro
    cells.append(nbf.v4.new_markdown_cell("""# CR_Score Playbook 01: Quick Start

**Level:** Beginner  
**Time:** 5-10 minutes  
**Goal:** Build your first credit scorecard in 3 lines of code

## What You'll Learn

- Load credit application data
- Build a complete scorecard in 3 lines
- Evaluate model performance
- Score new applications

## Prerequisites

- Python 3.9+
- CR_Score installed (`pip install -e .` from project root)
- No PySpark required!"""))
    
    # Step 1: Setup
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Setup and Load Data

First, let's import the necessary libraries and load our sample data."""))
    
    cells.append(nbf.v4.new_code_cell("""# Import libraries
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / 'src'))

# Import CR_Score
from cr_score import ScorecardPipeline

print("[OK] Libraries imported successfully!")"""))
    
    cells.append(nbf.v4.new_code_cell("""# Load training data
train_df = pd.read_csv('data/train.csv')

print(f"Training data: {len(train_df)} applications")
print(f"Default rate: {train_df['default'].mean()*100:.2f}%")
print(f"\\nFeatures: {train_df.shape[1]} columns")

# Show sample
train_df.head()"""))
    
    cells.append(nbf.v4.new_code_cell("""# Load test data
test_df = pd.read_csv('data/test.csv')

print(f"Test data: {len(test_df)} applications")
print(f"Default rate: {test_df['default'].mean()*100:.2f}%")"""))
    
    # Step 2: Build
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Build Scorecard in 3 Lines!

This is where the magic happens. CR_Score makes it incredibly simple to build a complete scorecard."""))
    
    cells.append(nbf.v4.new_code_cell("""# LINE 1: Create pipeline
pipeline = ScorecardPipeline(
    max_n_bins=5,        # Maximum 5 bins per feature
    pdo=20,              # Every 20 points, odds double
    base_score=600       # Score 600 = 2% default rate
)

# LINE 2: Train on data
pipeline.fit(train_df, target_col='default')

# LINE 3: Predict scores
scores = pipeline.predict(test_df)

print("[OK] Scorecard built and scores predicted!")"""))
    
    # Step 3: Results
    cells.append(nbf.v4.new_markdown_cell("""## Step 3: Understand the Results

Let's see what our scorecard produced."""))
    
    cells.append(nbf.v4.new_code_cell("""# Score statistics
print("Score Statistics:")
print(f"  Mean:   {scores.mean():.0f}")
print(f"  Median: {np.median(scores):.0f}")
print(f"  Min:    {scores.min():.0f}")
print(f"  Max:    {scores.max():.0f}")
print(f"  Std:    {scores.std():.0f}")"""))
    
    cells.append(nbf.v4.new_code_cell("""# Plot score distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Separate by default status
goods = scores[test_df['default'] == 0]
bads = scores[test_df['default'] == 1]

plt.hist(goods, bins=30, alpha=0.6, label='Good (no default)', color='green')
plt.hist(bads, bins=30, alpha=0.6, label='Bad (default)', color='red')

plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.title('Score Distribution by Default Status')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("Higher scores = Lower risk (good separation!)")"""))
    
    # Step 4: Evaluate
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Evaluate Performance

Let's see how good our scorecard is."""))
    
    cells.append(nbf.v4.new_code_cell("""# Evaluate model
metrics = pipeline.evaluate(test_df, target_col='default')

print("Model Performance:")
print(f"  AUC:   {metrics['auc']:.3f} {'(Excellent!)' if metrics['auc'] >= 0.8 else '(Good!)' if metrics['auc'] >= 0.7 else ''}")
print(f"  Gini:  {metrics['gini']:.3f}")
print(f"  KS:    {metrics['ks']:.3f}")
print("\\nInterpretation:")
print(f"  - AUC {metrics['auc']:.3f} means the model is {'excellent' if metrics['auc'] >= 0.8 else 'good' if metrics['auc'] >= 0.7 else 'fair'}")
print(f"  - It can distinguish between good and bad customers!")"""))
    
    # Step 5: Features
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: See Which Features Were Selected

Let's understand what the model is using."""))
    
    cells.append(nbf.v4.new_code_cell("""# Get pipeline summary
summary = pipeline.get_summary()

print(f"Number of features selected: {summary['n_features']}")
print(f"\\nSelected features:")
for i, feature in enumerate(summary['selected_features'], 1):
    print(f"  {i}. {feature}")"""))
    
    cells.append(nbf.v4.new_code_cell("""# See feature importance (IV values)
iv_df = pd.DataFrame(summary['iv_summary'])
iv_df = iv_df.sort_values('iv', ascending=False)

print("\\nFeature Importance (Information Value):")
print(iv_df.to_string(index=False))

print("\\nIV Interpretation:")
print("  < 0.02: Weak")
print("  0.02-0.1: Medium")
print("  0.1-0.3: Strong")
print("  > 0.3: Very Strong")"""))
    
    # Step 6: Score new
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Score New Applications

Now let's use our scorecard to score new loan applications."""))
    
    cells.append(nbf.v4.new_code_cell("""# Take first 10 applications from test set as "new" applications
new_applications = test_df.head(10).copy()

# Score them
new_scores = pipeline.predict(new_applications)
new_probas = pipeline.predict_proba(new_applications)

# Add to dataframe
new_applications['credit_score'] = new_scores
new_applications['default_probability'] = new_probas

# Show results
display_cols = ['application_id', 'age', 'income', 'debt_to_income_ratio', 
                'credit_score', 'default_probability', 'default']

print("New Application Scores:")
print(new_applications[display_cols].to_string(index=False))"""))
    
    cells.append(nbf.v4.new_code_cell("""# Make decisions based on scores
def make_decision(score):
    if score >= 650:
        return 'APPROVE'
    elif score >= 600:
        return 'REVIEW'
    else:
        return 'DECLINE'

new_applications['decision'] = new_applications['credit_score'].apply(make_decision)

print("\\nDecisions:")
decision_cols = ['application_id', 'credit_score', 'decision', 'default']
print(new_applications[decision_cols].to_string(index=False))

print("\\nDecision Summary:")
print(new_applications['decision'].value_counts())"""))
    
    # Step 7: Save
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Save Your Scorecard

Let's save the scorecard for production use."""))
    
    cells.append(nbf.v4.new_code_cell("""# Save pipeline
import pickle

with open('my_first_scorecard.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("[OK] Scorecard saved to 'my_first_scorecard.pkl'")

# You can load it later like this:
# with open('my_first_scorecard.pkl', 'rb') as f:
#     loaded_pipeline = pickle.load(f)
#     scores = loaded_pipeline.predict(new_data)"""))
    
    # Summary
    cells.append(nbf.v4.new_markdown_cell("""## Summary

Congratulations! You just built your first credit scorecard in 3 lines of code!

### What You Did:

1. Loaded credit application data
2. Built a complete scorecard in 3 lines
3. Evaluated performance (AUC, Gini, KS)
4. Understood which features are important
5. Scored new applications
6. Made approve/decline decisions
7. Saved the scorecard for production

### Next Steps:

- **Playbook 02**: Learn feature selection to pick the best features
- **Playbook 03**: Create beautiful visualizations and reports
- **Playbook 04**: Master the complete scorecard workflow
- **Playbook 05**: Explore advanced topics

### Key Takeaway:

CR_Score makes scorecard development **simple** without sacrificing **power**. You got enterprise-grade results with beginner-friendly code!"""))
    
    nb['cells'] = cells
    return nb


def main():
    """Generate all notebooks."""
    output_dir = Path(__file__).parent
    
    print("Generating CR_Score Playbook Notebooks...")
    print("=" * 60)
    
    # Generate Notebook 01
    print("[1/5] Creating 01_quickstart.ipynb...")
    nb01 = create_01_quickstart()
    nbf.write(nb01, output_dir / '01_quickstart.ipynb')
    print("  [OK] Playbook 01 created (Beginner, 5-10 min)")
    
    print("\\n" + "=" * 60)
    print("[OK] Notebooks generated successfully!")
    print("\\nTo run:")
    print("  cd playbooks")
    print("  jupyter notebook")
    print("\\nStart with: 01_quickstart.ipynb")


if __name__ == "__main__":
    main()
