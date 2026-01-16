Scaling (PDO Transformation)
=============================

Convert probabilities to credit scores using PDO (Points to Double Odds) transformation.

Overview
--------

The PDO (Points to Double Odds) scaling module transforms default probabilities into 
interpretable credit scores using the industry-standard PDO formula.

**What is PDO?**

PDO stands for "Points to Double Odds" - it defines how many points it takes for 
the odds of default to double.

**Key Concepts:**

- **Score**: A number (typically 300-850) that represents creditworthiness
- **Higher Score = Lower Risk**: Better customers have higher scores
- **PDO**: Points needed for odds to double (typically 20, 50, or 100)
- **Base Score**: Reference point (e.g., 600)
- **Base Odds**: Odds at the base score (e.g., 50:1 = 2% default rate)

Why Use PDO Scaling?
--------------------

✅ **Interpretability**: Scores are easier to understand than probabilities

- "Score 650" is clearer than "3.8% default probability"

✅ **Industry Standard**: FICO and other credit bureaus use PDO scoring

✅ **Business-Friendly**: Stakeholders understand scores better

✅ **Consistent Scale**: Same scale across different models

✅ **Decision Rules**: Easy to set cutoff scores (e.g., approve if score > 620)

PDOScaler Class
---------------

.. autoclass:: cr_score.scaling.pdo.PDOScaler
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from cr_score.scaling import PDOScaler
   import numpy as np

   # Create scaler
   scaler = PDOScaler(
       pdo=20,           # Every 20 points, odds double
       base_score=600,   # Reference score
       base_odds=50.0    # 2% default rate at base score
   )

   # Convert probabilities to scores
   probabilities = np.array([0.01, 0.02, 0.05, 0.10])
   scores = scaler.transform(probabilities)

   print(scores)
   # Output: [669.65, 640.00, 600.00, 560.00]

Understanding the Parameters
-----------------------------

PDO (Points to Double Odds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition:** Number of points needed for odds to double

**Common Values:**

- **20**: More granular, scores change quickly (most common)
- **50**: Moderate granularity
- **100**: Less granular, scores change slowly

**Example:**

.. code-block:: python

   # With PDO=20
   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)
   
   # At score 600: 2% default rate
   # At score 620 (+20 points): 1% default rate (odds doubled)
   # At score 640 (+40 points): 0.5% default rate (odds doubled again)

**Choosing PDO:**

- **Use 20**: Standard for most scorecards (FICO-style)
- **Use 50**: If you want slower score changes
- **Use 100**: If you want very stable scores

Base Score
~~~~~~~~~~

**Definition:** Reference point for the scoring scale

**Common Values:**

- **600**: FICO-style (most common)
- **500**: Some custom scorecards
- **700**: Premium/low-risk portfolios

**Example:**

.. code-block:: python

   # FICO-style scoring
   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

   # Custom scoring
   scaler = PDOScaler(pdo=20, base_score=500, base_odds=50)

**Choosing Base Score:**

- **Use 600**: Industry standard, familiar to stakeholders
- **Use 500**: If you want lower score range
- **Use 700**: If you want higher score range

Base Odds
~~~~~~~~~

**Definition:** Odds at the base score

**Formula:** odds = (1 - p) / p, where p is default probability

**Common Values:**

- **50**: 2% default rate (98:2 = 50:1)
- **19**: 5% default rate (95:5 = 19:1)
- **99**: 1% default rate (99:1)

**Example:**

.. code-block:: python

   # 2% default rate at score 600
   scaler = PDOScaler(base_score=600, base_odds=50, pdo=20)

   # 5% default rate at score 600
   scaler = PDOScaler(base_score=600, base_odds=19, pdo=20)

**Choosing Base Odds:**

- **Use 50** (2%): Conservative, typical for personal loans
- **Use 19** (5%): More aggressive, typical for subprime
- **Use 99** (1%): Very conservative, prime customers

Converting Between Probabilities and Odds
------------------------------------------

**Probability to Odds:**

.. code-block:: python

   def prob_to_odds(prob):
       return (1 - prob) / prob

   # Examples
   prob_to_odds(0.01)  # = 99 (1% default)
   prob_to_odds(0.02)  # = 49 (2% default)
   prob_to_odds(0.05)  # = 19 (5% default)
   prob_to_odds(0.10)  # = 9  (10% default)

**Odds to Probability:**

.. code-block:: python

   def odds_to_prob(odds):
       return 1 / (1 + odds)

   # Examples
   odds_to_prob(99)  # = 0.0101 (1% default)
   odds_to_prob(50)  # = 0.0196 (2% default)
   odds_to_prob(19)  # = 0.0500 (5% default)

Complete Example
----------------

Step-by-Step Scoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cr_score.scaling import PDOScaler
   from cr_score.model import LogisticScorecard
   import pandas as pd

   # 1. Train model (assume already done)
   model = LogisticScorecard()
   model.fit(X_train, y_train)

   # 2. Get probabilities
   probas = model.predict_proba(X_test)[:, 1]

   # 3. Create scaler
   scaler = PDOScaler(
       pdo=20,           # Industry standard
       base_score=600,   # FICO-style
       base_odds=50.0    # 2% default at 600
   )

   # 4. Convert to scores
   scores = scaler.transform(probas)

   # 5. Analyze results
   print(f"Score range: {scores.min():.0f} - {scores.max():.0f}")
   print(f"Mean score: {scores.mean():.0f}")
   print(f"Median score: {np.median(scores):.0f}")

   # 6. Create score bands
   df_results = pd.DataFrame({
       'probability': probas,
       'score': scores,
       'score_band': pd.cut(scores, bins=[0, 580, 620, 660, 700, 1000],
                            labels=['Very High Risk', 'High Risk', 'Medium Risk', 
                                    'Low Risk', 'Very Low Risk'])
   })

   print(df_results.groupby('score_band').size())

Inverse Transformation
~~~~~~~~~~~~~~~~~~~~~~

Convert scores back to probabilities:

.. code-block:: python

   # Transform probabilities to scores
   scores = scaler.transform(probabilities)

   # Transform scores back to probabilities
   probabilities_back = scaler.inverse_transform(scores)

   # Verify they match
   np.allclose(probabilities, probabilities_back)  # True

Common Use Cases
----------------

Use Case 1: Standard FICO-Style Scorecard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   scaler = PDOScaler(
       pdo=20,           # Standard
       base_score=600,   # FICO reference
       base_odds=50.0    # 2% at 600
   )

   # Interpretation:
   # - Score 600 = 2% default rate
   # - Score 620 = 1% default rate
   # - Score 640 = 0.5% default rate
   # - Score 580 = 4% default rate
   # - Score 560 = 8% default rate

Use Case 2: Subprime Lending
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   scaler = PDOScaler(
       pdo=20,
       base_score=600,
       base_odds=19.0    # 5% at 600 (higher risk tolerance)
   )

Use Case 3: Premium/Prime Customers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   scaler = PDOScaler(
       pdo=20,
       base_score=700,   # Higher reference
       base_odds=99.0    # 1% at 700
   )

Use Case 4: Fine-Grained Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   scaler = PDOScaler(
       pdo=50,           # Slower changes
       base_score=600,
       base_odds=50.0
   )

Score Interpretation Guide
---------------------------

For FICO-Style Scoring (PDO=20, base=600, odds=50)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Score Range    | Default Rate | Risk Level      | Decision
   ---------------|--------------|-----------------|------------------
   750+           | < 0.25%      | Excellent       | Auto-approve
   700-750        | 0.25-0.5%    | Very Good       | Approve
   650-700        | 0.5-1%       | Good            | Approve
   600-650        | 1-2%         | Fair            | Review
   550-600        | 2-4%         | Below Average   | Additional review
   500-550        | 4-8%         | Poor            | Likely decline
   < 500          | > 8%         | Very Poor       | Decline

Creating Score Bands
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   def create_score_bands(scores):
       """Create interpretable score bands"""
       bands = pd.cut(
           scores,
           bins=[0, 550, 600, 650, 700, 1000],
           labels=['Very High Risk', 'High Risk', 'Medium Risk', 
                   'Low Risk', 'Very Low Risk']
       )
       return bands

   # Apply to scores
   score_bands = create_score_bands(scores)

   # Analyze distribution
   print(score_bands.value_counts())

Integration with Pipeline
--------------------------

The PDOScaler is automatically used in :class:`~cr_score.pipeline.ScorecardPipeline`:

.. code-block:: python

   from cr_score import ScorecardPipeline

   pipeline = ScorecardPipeline(
       pdo=20,           # Passed to PDOScaler
       base_score=600,   # Passed to PDOScaler
       base_odds=50.0    # Passed to PDOScaler
   )

   pipeline.fit(train_df, target_col="default")

   # Pipeline automatically:
   # 1. Trains model
   # 2. Gets probabilities
   # 3. Converts to scores using PDOScaler

   scores = pipeline.predict(test_df)

Mathematical Details
--------------------

PDO Formula
~~~~~~~~~~~

The PDO transformation formula is:

.. math::

   Score = offset - factor \\times \\log(odds)

Where:

.. math::

   factor = \\frac{PDO}{\\log(2)}

.. math::

   offset = base\\_score - factor \\times \\log(base\\_odds)

.. math::

   odds = \\frac{1 - p}{p}

**Example Calculation:**

For PDO=20, base_score=600, base_odds=50, probability=0.02:

.. code-block:: python

   import numpy as np

   pdo = 20
   base_score = 600
   base_odds = 50
   probability = 0.02

   # Calculate factor
   factor = pdo / np.log(2)  # = 28.85

   # Calculate offset
   offset = base_score - factor * np.log(base_odds)  # = 487.12

   # Calculate odds from probability
   odds = (1 - probability) / probability  # = 49

   # Calculate score
   score = offset - factor * np.log(odds)  # ≈ 600

Properties and Validation
--------------------------

Score Range Validation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PDOScaler ensures valid scores
   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

   # Probabilities must be in (0, 1)
   valid_probs = np.array([0.001, 0.01, 0.1, 0.5])
   scores = scaler.transform(valid_probs)

   # Invalid probabilities will raise error
   try:
       invalid_probs = np.array([0.0, 1.0])  # Boundary values
       scaler.transform(invalid_probs)
   except ValueError:
       print("Invalid probabilities")

Score Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

   # Test key properties
   prob_at_base = 1 / (1 + scaler.base_odds)
   score_at_base = scaler.transform([prob_at_base])[0]

   print(f"At base score {scaler.base_score}:")
   print(f"  Probability: {prob_at_base:.4f}")
   print(f"  Score: {score_at_base:.2f}")

   # Test PDO property
   prob_double_odds = 1 / (1 + 2 * scaler.base_odds)
   score_double_odds = scaler.transform([prob_double_odds])[0]

   print(f"\nAt double odds (half probability):")
   print(f"  Probability: {prob_double_odds:.4f}")
   print(f"  Score: {score_double_odds:.2f}")
   print(f"  Difference: {score_double_odds - score_at_base:.2f} (should be {scaler.pdo})")

Exporting Scorecard
-------------------

Export scoring parameters for production:

.. code-block:: python

   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

   # Get parameters
   params = {
       'pdo': scaler.pdo,
       'base_score': scaler.base_score,
       'base_odds': scaler.base_odds,
       'factor': scaler.factor,
       'offset': scaler.offset
   }

   # Save to file
   import json
   with open('scoring_params.json', 'w') as f:
       json.dump(params, f, indent=2)

   # In production, load and use
   with open('scoring_params.json', 'r') as f:
       params = json.load(f)

   production_scaler = PDOScaler(
       pdo=params['pdo'],
       base_score=params['base_score'],
       base_odds=params['base_odds']
   )

Best Practices
--------------

1. Use Industry Standards
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Recommended for most scorecards
   scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)

2. Set Base Odds Based on Portfolio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate base odds from your target rate
   target_default_rate = 0.03  # 3%
   base_odds = (1 - target_default_rate) / target_default_rate

   scaler = PDOScaler(base_score=600, base_odds=base_odds, pdo=20)

3. Validate Score Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   scores = scaler.transform(probabilities)

   # Check distribution
   print(f"Min score: {scores.min():.0f}")
   print(f"Max score: {scores.max():.0f}")
   print(f"Mean score: {scores.mean():.0f}")
   print(f"Std score: {scores.std():.0f}")

   # Ensure reasonable range (typically 300-850)
   assert scores.min() >= 300, "Scores too low"
   assert scores.max() <= 850, "Scores too high"

4. Document Your Scoring Scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Always document what your scores mean
   scoring_documentation = f"""
   Scoring Scale:
   - PDO: {scaler.pdo}
   - Base Score: {scaler.base_score}
   - At score {scaler.base_score}: {1/(1+scaler.base_odds):.2%} default rate
   - Every {scaler.pdo} points: odds double
   """

   print(scoring_documentation)

Common Issues and Solutions
----------------------------

Issue 1: Scores Out of Expected Range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Scores are < 300 or > 850

**Solution:** Adjust base_score and base_odds

.. code-block:: python

   # If scores too high, decrease base_score or increase base_odds
   scaler = PDOScaler(base_score=550, base_odds=50, pdo=20)

   # If scores too low, increase base_score or decrease base_odds
   scaler = PDOScaler(base_score=650, base_odds=50, pdo=20)

Issue 2: Scores Too Sensitive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Small probability changes cause large score changes

**Solution:** Increase PDO

.. code-block:: python

   # More stable scores
   scaler = PDOScaler(pdo=50, base_score=600, base_odds=50)

Issue 3: Stakeholders Don't Understand Scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution:** Provide interpretation guide

.. code-block:: python

   def explain_score(score, scaler):
       """Explain what a score means"""
       prob = scaler.inverse_transform([score])[0]

       return f"""
       Score: {score:.0f}
       Default Probability: {prob:.2%}
       Interpretation: {
           'Excellent' if score >= 700 else
           'Good' if score >= 650 else
           'Fair' if score >= 600 else
           'Below Average' if score >= 550 else
           'Poor'
       }
       """

   print(explain_score(680, scaler))

See Also
--------

- :doc:`/api/pipeline` - ScorecardPipeline uses PDOScaler automatically
- :doc:`/api/calibration` - Calibration before scaling
- :doc:`/api/model` - Model that produces probabilities
- :doc:`/guides/quickstart` - Complete scorecard example
