"""
Feature importance analysis for scorecards.

Provides multiple methods for analyzing feature importance
including permutation importance, drop-column importance, and correlation analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from cr_score.core.logging import get_audit_logger


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods.
    
    Provides comprehensive feature importance analysis including:
    - Model coefficients
    - Permutation importance
    - Drop-column importance
    - Information Value (IV)
    
    Example:
        >>> analyzer = FeatureImportanceAnalyzer(model)
        >>> importance = analyzer.analyze(X_test, y_test)
        >>> analyzer.plot_importance(importance)
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize feature importance analyzer.
        
        Args:
            model: Fitted model
            feature_names: Feature names (auto-detected if None)
        """
        self.model = model
        self.feature_names = feature_names
        self.logger = get_audit_logger()
    
    def get_coefficient_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model coefficients.
        
        Returns:
            DataFrame with features and coefficients
        """
        if not hasattr(self.model, 'get_coefficients'):
            raise ValueError("Model does not have get_coefficients method")
        
        coefs = self.model.get_coefficients()
        coefs['abs_importance'] = coefs['abs_coefficient']
        
        return coefs[['feature', 'coefficient', 'abs_importance']].sort_values(
            'abs_importance',
            ascending=False
        )
    
    def get_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: str = 'roc_auc',
    ) -> pd.DataFrame:
        """
        Calculate permutation importance.
        
        Shuffles each feature and measures performance drop.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            scoring: Scoring metric
        
        Returns:
            DataFrame with permutation importance
        """
        self.logger.info(
            "Calculating permutation importance",
            n_features=X.shape[1],
            n_repeats=n_repeats,
        )
        
        result = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
        )
        
        df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values('importance_mean', ascending=False)
        
        return df
    
    def get_drop_column_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric_func: Any = roc_auc_score,
    ) -> pd.DataFrame:
        """
        Calculate drop-column importance.
        
        Measures performance drop when each feature is removed.
        
        Args:
            X: Feature matrix
            y: Target variable
            metric_func: Metric function to use
        
        Returns:
            DataFrame with drop-column importance
        """
        self.logger.info(
            "Calculating drop-column importance",
            n_features=X.shape[1],
        )
        
        # Baseline performance
        y_pred = self.model.predict_proba(X)[:, 1]
        baseline_score = metric_func(y, y_pred)
        
        importance_scores = []
        
        for col in X.columns:
            # Drop column and evaluate
            X_dropped = X.drop(columns=[col])
            
            # Retrain model (if possible) or just predict without feature
            # For now, we'll estimate impact by setting to mean
            X_temp = X.copy()
            X_temp[col] = X_temp[col].mean()
            
            y_pred_temp = self.model.predict_proba(X_temp)[:, 1]
            dropped_score = metric_func(y, y_pred_temp)
            
            importance = baseline_score - dropped_score
            importance_scores.append({
                'feature': col,
                'importance': importance,
                'baseline_score': baseline_score,
                'dropped_score': dropped_score,
            })
        
        df = pd.DataFrame(importance_scores).sort_values(
            'importance',
            ascending=False
        )
        
        return df
    
    def get_correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculate feature-target correlations.
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            DataFrame with correlations
        """
        correlations = []
        
        for col in X.columns:
            corr = X[col].corr(y)
            correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr),
            })
        
        df = pd.DataFrame(correlations).sort_values(
            'abs_correlation',
            ascending=False
        )
        
        return df
    
    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive feature importance analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: Methods to use (None = all available)
        
        Returns:
            Dictionary of importance DataFrames by method
        """
        if methods is None:
            methods = ['coefficient', 'permutation', 'correlation']
        
        results = {}
        
        if 'coefficient' in methods:
            try:
                results['coefficient'] = self.get_coefficient_importance()
            except ValueError:
                self.logger.warning("Coefficient importance not available")
        
        if 'permutation' in methods:
            results['permutation'] = self.get_permutation_importance(X, y)
        
        if 'drop_column' in methods:
            results['drop_column'] = self.get_drop_column_importance(X, y)
        
        if 'correlation' in methods:
            results['correlation'] = self.get_correlation_importance(X, y)
        
        return results
    
    def get_consensus_importance(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        method: str = "rank_average",
    ) -> pd.DataFrame:
        """
        Combine multiple importance measures into consensus ranking.
        
        Args:
            importance_dict: Dictionary of importance DataFrames
            method: Combination method ('rank_average', 'score_average')
        
        Returns:
            DataFrame with consensus importance
        """
        if method == "rank_average":
            # Average the ranks across methods
            all_ranks = []
            
            for name, df in importance_dict.items():
                ranks = df.copy()
                ranks['rank'] = range(1, len(ranks) + 1)
                ranks = ranks[['feature', 'rank']].rename(
                    columns={'rank': f'rank_{name}'}
                )
                all_ranks.append(ranks)
            
            # Merge all ranks
            consensus = all_ranks[0]
            for ranks in all_ranks[1:]:
                consensus = consensus.merge(ranks, on='feature', how='outer')
            
            # Calculate average rank
            rank_cols = [c for c in consensus.columns if c.startswith('rank_')]
            consensus['avg_rank'] = consensus[rank_cols].mean(axis=1)
            consensus['consensus_score'] = 1 / consensus['avg_rank']
            
            consensus = consensus.sort_values('avg_rank')
        
        elif method == "score_average":
            # Normalize and average scores
            all_scores = []
            
            for name, df in importance_dict.items():
                scores = df.copy()
                importance_col = [c for c in scores.columns if 'importance' in c or 'coefficient' in c][0]
                
                # Normalize to 0-1
                scores['normalized'] = (
                    (scores[importance_col] - scores[importance_col].min()) /
                    (scores[importance_col].max() - scores[importance_col].min())
                )
                
                scores = scores[['feature', 'normalized']].rename(
                    columns={'normalized': f'score_{name}'}
                )
                all_scores.append(scores)
            
            # Merge all scores
            consensus = all_scores[0]
            for scores in all_scores[1:]:
                consensus = consensus.merge(scores, on='feature', how='outer')
            
            # Calculate average score
            score_cols = [c for c in consensus.columns if c.startswith('score_')]
            consensus['consensus_score'] = consensus[score_cols].mean(axis=1)
            consensus = consensus.sort_values('consensus_score', ascending=False)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return consensus
    
    def plot_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to plot
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        # Get importance column
        imp_col = [c for c in importance_df.columns 
                   if 'importance' in c or 'score' in c][0]
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(top_features)),
            top_features[imp_col],
            color='skyblue'
        )
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def export_importance(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        filepath: str,
    ) -> None:
        """
        Export importance analysis to Excel.
        
        Args:
            importance_dict: Dictionary of importance DataFrames
            filepath: Output file path
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for name, df in importance_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)
        
        self.logger.info(f"Feature importance exported to {filepath}")
