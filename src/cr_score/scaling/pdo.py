"""
PDO (Points-Double-Odds) scaling for credit scores.

Industry standard transformation from probability to credit score.
"""

from typing import Dict

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class PDOScaler:
    """
    PDO (Points-Double-Odds) scaler for credit scores.

    Transforms probabilities to scores using industry standard formula:
        Score = Offset + Factor * log(Odds)

    Where:
        Factor = PDO / log(2)
        Offset = BaseScore - Factor * log(BaseOdds)

    Example:
        >>> scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)
        >>> scores = scaler.transform(probabilities)
        >>> # Score of 600 means 50:1 odds (2% default probability)
        >>> # Every 20 points, odds double (score 620 = 100:1 odds = 1% default)
    """

    def __init__(
        self,
        pdo: int = 20,
        base_score: int = 600,
        base_odds: float = 50.0,
    ) -> None:
        """
        Initialize PDO scaler.

        Args:
            pdo: Points to double the odds (typical: 20)
            base_score: Score at base odds (typical: 600)
            base_odds: Odds at base score (typical: 50 = 2% default rate)

        Example:
            >>> # Score 600 = 50:1 odds = 2% default
            >>> # Score 620 = 100:1 odds = 1% default  
            >>> # Score 580 = 25:1 odds = 4% default
            >>> scaler = PDOScaler(pdo=20, base_score=600, base_odds=50)
        """
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.logger = get_audit_logger()

        # Calculate factor and offset
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.logger.info(
            "PDO scaler initialized",
            pdo=pdo,
            base_score=base_score,
            base_odds=base_odds,
            factor=self.factor,
            offset=self.offset,
        )

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Transform probabilities to scores.

        Args:
            probabilities: Default probabilities (range 0-1)

        Returns:
            Credit scores

        Example:
            >>> scores = scaler.transform(df["default_probability"])
            >>> print(f"Mean score: {scores.mean():.0f}")
        """
        # Convert probability to odds
        # Odds = (1 - P) / P  (good odds, not bad odds)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        odds = (1 - probabilities) / probabilities

        # Apply PDO formula
        scores = self.offset + self.factor * np.log(odds)

        return scores

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores back to probabilities.

        Args:
            scores: Credit scores

        Returns:
            Default probabilities

        Example:
            >>> probabilities = scaler.inverse_transform(np.array([600, 620, 580]))
            >>> print(probabilities)  # [0.02, 0.01, 0.04]
        """
        # Reverse PDO formula
        log_odds = (scores - self.offset) / self.factor
        odds = np.exp(log_odds)

        # Convert odds to probability
        probabilities = 1 / (1 + odds)

        return probabilities

    def get_score_at_odds(self, odds: float) -> float:
        """
        Calculate score for given odds.

        Args:
            odds: Odds ratio (good:bad)

        Returns:
            Credit score

        Example:
            >>> score_at_25_to_1 = scaler.get_score_at_odds(25.0)
            >>> print(f"Score: {score_at_25_to_1:.0f}")  # 580
        """
        return self.offset + self.factor * np.log(odds)

    def get_odds_at_score(self, score: float) -> float:
        """
        Calculate odds for given score.

        Args:
            score: Credit score

        Returns:
            Odds ratio

        Example:
            >>> odds = scaler.get_odds_at_score(600)
            >>> print(f"Odds: {odds:.1f}:1")  # 50.0:1
        """
        log_odds = (score - self.offset) / self.factor
        return np.exp(log_odds)

    def get_probability_at_score(self, score: float) -> float:
        """
        Calculate default probability for given score.

        Args:
            score: Credit score

        Returns:
            Default probability

        Example:
            >>> prob = scaler.get_probability_at_score(600)
            >>> print(f"Default probability: {prob:.2%}")  # 2.00%
        """
        odds = self.get_odds_at_score(score)
        return 1 / (1 + odds)

    def create_score_bands(
        self,
        n_bands: int = 10,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Create score band definitions.

        Args:
            n_bands: Number of score bands
            min_score: Minimum score (auto if None)
            max_score: Maximum score (auto if None)

        Returns:
            DataFrame with score bands, odds, and probabilities

        Example:
            >>> bands = scaler.create_score_bands(n_bands=10)
            >>> print(bands[["band", "score_min", "score_max", "default_probability"]])
        """
        if min_score is None:
            min_score = int(self.base_score - 5 * self.pdo)

        if max_score is None:
            max_score = int(self.base_score + 5 * self.pdo)

        # Create bands
        band_edges = np.linspace(min_score, max_score, n_bands + 1)

        bands = []
        for i in range(n_bands):
            band_min = band_edges[i]
            band_max = band_edges[i + 1]
            band_mid = (band_min + band_max) / 2

            bands.append({
                "band": i + 1,
                "band_label": f"Band {i+1}",
                "score_min": int(band_min),
                "score_max": int(band_max),
                "score_midpoint": int(band_mid),
                "odds": self.get_odds_at_score(band_mid),
                "default_probability": self.get_probability_at_score(band_mid),
            })

        return pd.DataFrame(bands)

    def export_scaling_spec(self) -> Dict:
        """
        Export scaling specification.

        Returns:
            Dictionary with PDO parameters

        Example:
            >>> spec = scaler.export_scaling_spec()
            >>> import json
            >>> with open("scaling_spec.json", "w") as f:
            ...     json.dump(spec, f)
        """
        return {
            "pdo": self.pdo,
            "base_score": self.base_score,
            "base_odds": self.base_odds,
            "factor": float(self.factor),
            "offset": float(self.offset),
            "formula": "Score = Offset + Factor * log(Odds)",
            "interpretation": {
                "base_score_meaning": f"Score of {self.base_score} means {self.base_odds}:1 odds ({self.get_probability_at_score(self.base_score):.2%} default)",
                "pdo_meaning": f"Every {self.pdo} points, odds double",
            },
        }
