"""
High-level scorecard pipeline for simplified workflows.

Provides a scikit-learn-like interface for complete scorecard development.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from cr_score.binning.optbinning_wrapper import AutoBinner
from cr_score.model import LogisticScorecard
from cr_score.calibration import InterceptCalibrator
from cr_score.scaling import PDOScaler
from cr_score.features.selection import ForwardSelector, BackwardSelector, StepwiseSelector
from cr_score.core.logging import get_audit_logger


class ScorecardPipeline:
    """
    Complete scorecard pipeline with automatic binning and WoE encoding.

    Simplifies scorecard development to a few lines of code using optimal binning.

    Example:
        >>> # Simple 3-line scorecard development
        >>> pipeline = ScorecardPipeline()
        >>> pipeline.fit(df_train, target_col="default")
        >>> scores = pipeline.predict(df_test)

    Example (with configuration):
        >>> pipeline = ScorecardPipeline(
        ...     max_n_bins=5,
        ...     pdo=20,
        ...     base_score=600,
        ...     target_bad_rate=0.05
        ... )
        >>> pipeline.fit(df_train, target_col="default")
        >>> scores = pipeline.predict(df_test)
        >>> print(pipeline.get_summary())
    """

    def __init__(
        self,
        max_n_bins: int = 5,
        min_iv: float = 0.02,
        pdo: int = 20,
        base_score: int = 600,
        base_odds: float = 50.0,
        target_bad_rate: Optional[float] = None,
        calibrate: bool = True,
        feature_selection: Optional[str] = None,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize scorecard pipeline.

        Args:
            max_n_bins: Maximum bins per feature
            min_iv: Minimum IV to include feature
            pdo: Points to double odds
            base_score: Score at base odds
            base_odds: Odds at base score
            target_bad_rate: Target bad rate for calibration (None = use observed)
            calibrate: Whether to calibrate intercept
            feature_selection: Selection method (None, "forward", "backward", "stepwise")
            max_features: Maximum features to select (None = no limit)
            random_state: Random seed

        Example:
            >>> pipeline = ScorecardPipeline(
            ...     max_n_bins=5,
            ...     feature_selection="stepwise",
            ...     max_features=10
            ... )
        """
        self.max_n_bins = max_n_bins
        self.min_iv = min_iv
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.target_bad_rate = target_bad_rate
        self.calibrate = calibrate
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.random_state = random_state
        self.logger = get_audit_logger()

        # Components
        self.auto_binner_: Optional[AutoBinner] = None
        self.feature_selector_: Optional[Any] = None
        self.model_: Optional[LogisticScorecard] = None
        self.calibrator_: Optional[InterceptCalibrator] = None
        self.scaler_: Optional[PDOScaler] = None

        # State
        self.target_col_: Optional[str] = None
        self.feature_cols_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> "ScorecardPipeline":
        """
        Fit complete scorecard pipeline.

        Performs: Auto-binning → WoE encoding → Logistic regression → Calibration → Scaling

        Args:
            df: Training DataFrame
            target_col: Target variable column
            feature_cols: Features to use (None = all except target)
            sample_weight: Sample weights (for compressed data)

        Returns:
            Self

        Example:
            >>> pipeline.fit(df_train, target_col="default")
            >>> # That's it! Pipeline is ready to score.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting CR_Score Pipeline")
        self.logger.info("=" * 80)

        self.target_col_ = target_col

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]

        self.feature_cols_ = feature_cols

        # Step 1: Auto-binning with optimal algorithms
        self.logger.info("\n[1/5] Auto-binning with OptimalBinning...")
        self.auto_binner_ = AutoBinner(
            max_n_bins=self.max_n_bins,
            min_iv=self.min_iv,
        )

        _, df_woe = self.auto_binner_.fit_transform(df, target_col, feature_cols)

        selected_features = self.auto_binner_.get_selected_features()
        iv_summary = self.auto_binner_.get_iv_summary()

        self.logger.info(f"Selected {len(selected_features)} features with IV >= {self.min_iv}")
        self.logger.info("\nTop 5 features by IV:")
        for _, row in iv_summary.head(5).iterrows():
            self.logger.info(f"  {row['feature']}: IV={row['iv']:.3f}")

        if len(selected_features) == 0:
            raise ValueError("No features selected. Try lowering min_iv parameter.")

        # Step 2: Feature Selection (optional)
        woe_features = [f"{f}_woe" for f in selected_features]

        if self.feature_selection and len(selected_features) > 5:
            self.logger.info(f"\n[2/5] Running {self.feature_selection} feature selection...")

            # Create selector
            if self.feature_selection == "forward":
                self.feature_selector_ = ForwardSelector(
                    estimator=LogisticScorecard(random_state=self.random_state),
                    max_features=self.max_features,
                    scoring="roc_auc",
                    cv=3,
                    use_mlflow=False,
                )
            elif self.feature_selection == "backward":
                self.feature_selector_ = BackwardSelector(
                    estimator=LogisticScorecard(random_state=self.random_state),
                    min_features=3,
                    scoring="roc_auc",
                    cv=3,
                    use_mlflow=False,
                )
            elif self.feature_selection == "stepwise":
                self.feature_selector_ = StepwiseSelector(
                    estimator=LogisticScorecard(random_state=self.random_state),
                    max_features=self.max_features,
                    scoring="roc_auc",
                    cv=3,
                    use_mlflow=False,
                )
            else:
                self.logger.warning(f"Unknown selection method: {self.feature_selection}, skipping")
                self.feature_selector_ = None

            if self.feature_selector_:
                self.feature_selector_.fit(df_woe[woe_features], df[target_col])
                woe_features = self.feature_selector_.get_selected_features()

                # Map back to original features
                self.selected_features_ = [f.replace("_woe", "") for f in woe_features]

                self.logger.info(
                    f"{self.feature_selection.capitalize()} selection: "
                    f"{len(selected_features)} → {len(self.selected_features_)} features"
                )
            else:
                self.selected_features_ = selected_features
        else:
            self.logger.info("\n[2/5] Skipping feature selection (not configured or too few features)")
            self.selected_features_ = selected_features

        # Step 3: Model training
        self.logger.info(f"\n[3/5] Training logistic regression with {len(self.selected_features_)} features...")
        self.model_ = LogisticScorecard(random_state=self.random_state)

        X_woe = df_woe[woe_features]
        y = df[target_col]

        self.model_.fit(X_woe, y, sample_weight=sample_weight)

        # Evaluate on training set
        train_proba = self.model_.predict_proba(X_woe)[:, 1]
        train_metrics = self.model_.get_performance_metrics(y, train_proba)

        self.logger.info(f"Training AUC: {train_metrics['auc']:.3f}")
        self.logger.info(f"Training Gini: {train_metrics['gini']:.3f}")
        self.logger.info(f"Training KS: {train_metrics['ks']:.3f}")

        # Step 4: Calibration
        if self.calibrate:
            self.logger.info("\n[4/5] Calibrating intercept...")
            self.calibrator_ = InterceptCalibrator(target_bad_rate=self.target_bad_rate)
            self.calibrator_.fit(train_proba, y)

            train_proba_calibrated = self.calibrator_.transform(train_proba)
            self.logger.info(f"Target bad rate: {self.calibrator_.target_bad_rate:.2%}")
            self.logger.info(f"Achieved bad rate: {train_proba_calibrated.mean():.2%}")
        else:
            self.logger.info("\n[4/5] Skipping calibration (calibrate=False)")
            train_proba_calibrated = train_proba

        # Step 5: Scaling
        self.logger.info("\n[5/5] Scaling to credit scores...")
        self.scaler_ = PDOScaler(
            pdo=self.pdo,
            base_score=self.base_score,
            base_odds=self.base_odds,
        )

        train_scores = self.scaler_.transform(train_proba_calibrated)

        self.logger.info(f"Score range: {train_scores.min():.0f} - {train_scores.max():.0f}")
        self.logger.info(f"Mean score: {train_scores.mean():.0f}")

        # Done
        self.logger.info("\n✓ Pipeline fitted successfully!")
        self.logger.info("=" * 80)

        self.is_fitted_ = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict credit scores.

        Args:
            df: DataFrame with features

        Returns:
            Array of credit scores

        Example:
            >>> scores = pipeline.predict(df_test)
            >>> print(f"Mean score: {scores.mean():.0f}")
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Transform to WoE
        df_woe = self.auto_binner_.transform_woe(df)

        # Use selected features (from feature selection or auto-binning)
        features_to_use = self.selected_features_ if self.selected_features_ else self.auto_binner_.get_selected_features()
        X_woe = df_woe[[f"{f}_woe" for f in features_to_use]]

        # Predict probabilities
        probas = self.model_.predict_proba(X_woe)[:, 1]

        # Calibrate
        if self.calibrate and self.calibrator_:
            probas = self.calibrator_.transform(probas)

        # Scale to scores
        scores = self.scaler_.transform(probas)

        return scores

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities.

        Args:
            df: DataFrame with features

        Returns:
            Array of default probabilities

        Example:
            >>> probas = pipeline.predict_proba(df_test)
            >>> print(f"Mean probability: {probas.mean():.2%}")
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Transform to WoE
        df_woe = self.auto_binner_.transform_woe(df)

        # Use selected features (from feature selection or auto-binning)
        features_to_use = self.selected_features_ if self.selected_features_ else self.auto_binner_.get_selected_features()
        X_woe = df_woe[[f"{f}_woe" for f in features_to_use]]

        # Predict probabilities
        probas = self.model_.predict_proba(X_woe)[:, 1]

        # Calibrate
        if self.calibrate and self.calibrator_:
            probas = self.calibrator_.transform(probas)

        return probas

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline on test set.

        Args:
            df: Test DataFrame
            target_col: Target column (None = use training target)

        Returns:
            Dictionary with performance metrics

        Example:
            >>> metrics = pipeline.evaluate(df_test)
            >>> print(f"Test AUC: {metrics['auc']:.3f}")
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted")

        if target_col is None:
            target_col = self.target_col_

        # Predict
        probas = self.predict_proba(df)
        scores = self.predict(df)

        # Evaluate
        y_true = df[target_col]
        metrics = self.model_.get_performance_metrics(y_true, probas)

        # Add score statistics
        metrics["score_min"] = float(scores.min())
        metrics["score_max"] = float(scores.max())
        metrics["score_mean"] = float(scores.mean())
        metrics["score_std"] = float(scores.std())

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """
        Get pipeline summary.

        Returns:
            Dictionary with pipeline configuration and feature importance

        Example:
            >>> summary = pipeline.get_summary()
            >>> print(summary["n_features"])
            >>> print(summary["iv_summary"])
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted")

        features_to_use = self.selected_features_ if self.selected_features_ else self.auto_binner_.get_selected_features()
        iv_summary = self.auto_binner_.get_iv_summary()

        # Filter IV summary to selected features only
        if self.selected_features_:
            iv_summary = iv_summary[iv_summary["feature"].isin(self.selected_features_)]

        summary = {
            "n_features": len(features_to_use),
            "selected_features": features_to_use,
            "iv_summary": iv_summary.to_dict("records"),
            "model_coefficients": self.model_.get_coefficients().to_dict("records"),
            "pdo_params": {
                "pdo": self.pdo,
                "base_score": self.base_score,
                "base_odds": self.base_odds,
            },
            "calibrated": self.calibrate,
        }

        if self.feature_selection:
            summary["feature_selection_method"] = self.feature_selection
            summary["features_before_selection"] = len(self.auto_binner_.get_selected_features())
            summary["features_after_selection"] = len(self.selected_features_) if self.selected_features_ else 0

        return summary

    def export_scorecard(self, output_path: str) -> None:
        """
        Export complete scorecard specification to JSON.

        Args:
            output_path: Output file path

        Example:
            >>> pipeline.export_scorecard("scorecard.json")
        """
        import json

        if not self.is_fitted_:
            raise ValueError("Pipeline not fitted")

        scorecard_spec = {
            "version": "1.2.0",
            "binning": {},
            "model": self.model_.export_model(),
            "scaling": self.scaler_.export_scaling_spec(),
            "features": self.auto_binner_.get_selected_features(),
            "iv_summary": self.auto_binner_.get_iv_summary().to_dict("records"),
        }

        # Export binning tables
        for feature in self.auto_binner_.get_selected_features():
            binner = self.auto_binner_.binners_[feature]
            binning_table = binner.get_binning_table()
            scorecard_spec["binning"][feature] = binning_table.to_dict("records")

        with open(output_path, "w") as f:
            json.dump(scorecard_spec, f, indent=2)

        self.logger.info(f"Scorecard exported to {output_path}")
