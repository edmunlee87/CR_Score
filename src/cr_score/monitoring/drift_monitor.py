"""
Data drift monitoring for production scorecards.

Detects distribution shifts in features and predictions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from cr_score.core.logging import get_audit_logger


class DriftMonitor:
    """
    Monitor data drift in production.
    
    Detects shifts in feature distributions using multiple statistical tests.
    
    Example:
        >>> monitor = DriftMonitor(reference_data=X_train)
        >>> drift_report = monitor.detect_drift(X_production)
        >>> monitor.plot_drift_summary(drift_report)
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        psi_threshold: float = 0.1,
        ks_threshold: float = 0.05,
        storage_path: Optional[str] = None,
    ) -> None:
        """
        Initialize drift monitor.
        
        Args:
            reference_data: Reference dataset (training data)
            psi_threshold: PSI alert threshold (0.1 = warning, 0.25 = critical)
            ks_threshold: KS test p-value threshold
            storage_path: Path to store drift data
        """
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.storage_path = Path(storage_path) if storage_path else Path('./monitoring_data')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_audit_logger()
        self.drift_history: List[Dict[str, Any]] = []
        
        # Calculate reference statistics
        if reference_data is not None:
            self.reference_stats = self._calculate_statistics(reference_data)
        else:
            self.reference_stats = {}
    
    def _calculate_statistics(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each feature."""
        stats_dict = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median()),
                    'q25': float(data[col].quantile(0.25)),
                    'q75': float(data[col].quantile(0.75)),
                }
            else:
                value_counts = data[col].value_counts(normalize=True)
                stats_dict[col] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'n_unique': int(data[col].nunique()),
                }
        
        return stats_dict
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for numeric features
        
        Returns:
            PSI value
        """
        # Create bins from reference
        if pd.api.types.is_numeric_dtype(reference):
            bins = pd.qcut(reference, q=n_bins, labels=False, duplicates='drop')
            bin_edges = pd.qcut(reference, q=n_bins, retbins=True, duplicates='drop')[1]
            
            # Apply same bins to current
            current_binned = pd.cut(
                current,
                bins=bin_edges,
                labels=False,
                include_lowest=True
            )
            
            # Calculate distributions
            ref_dist = bins.value_counts(normalize=True)
            cur_dist = current_binned.value_counts(normalize=True)
        else:
            # Categorical feature
            ref_dist = reference.value_counts(normalize=True)
            cur_dist = current.value_counts(normalize=True)
        
        # Align distributions
        all_categories = ref_dist.index.union(cur_dist.index)
        ref_dist = ref_dist.reindex(all_categories, fill_value=0.0001)
        cur_dist = cur_dist.reindex(all_categories, fill_value=0.0001)
        
        # Calculate PSI
        psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        
        return float(psi)
    
    def calculate_ks_statistic(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            reference: Reference distribution
            current: Current distribution
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        if pd.api.types.is_numeric_dtype(reference):
            statistic, pvalue = stats.ks_2samp(reference, current)
            return float(statistic), float(pvalue)
        else:
            return 0.0, 1.0  # Not applicable for categorical
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift in current data.
        
        Args:
            current_data: Current production data
            features: Features to check (None = all)
        
        Returns:
            Drift report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        timestamp = datetime.now().isoformat()
        
        if features is None:
            features = [col for col in current_data.columns 
                       if col in self.reference_data.columns]
        
        self.logger.info(
            "Detecting drift",
            n_features=len(features),
            n_samples=len(current_data),
        )
        
        drift_results = []
        
        for feature in features:
            ref_col = self.reference_data[feature]
            cur_col = current_data[feature]
            
            # Calculate PSI
            psi = self.calculate_psi(ref_col, cur_col)
            
            # Calculate KS statistic for numeric features
            ks_stat, ks_pval = self.calculate_ks_statistic(ref_col, cur_col)
            
            # Determine drift level
            if psi < self.psi_threshold:
                drift_level = 'stable'
            elif psi < self.psi_threshold * 2.5:
                drift_level = 'warning'
            else:
                drift_level = 'critical'
            
            # Calculate current statistics
            if pd.api.types.is_numeric_dtype(cur_col):
                cur_mean = float(cur_col.mean())
                ref_mean = self.reference_stats[feature]['mean']
                mean_change_pct = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
            else:
                mean_change_pct = None
            
            result = {
                'feature': feature,
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'drift_level': drift_level,
                'mean_change_pct': mean_change_pct,
            }
            
            drift_results.append(result)
        
        # Overall drift summary
        n_critical = sum(1 for r in drift_results if r['drift_level'] == 'critical')
        n_warning = sum(1 for r in drift_results if r['drift_level'] == 'warning')
        n_stable = sum(1 for r in drift_results if r['drift_level'] == 'stable')
        
        report = {
            'timestamp': timestamp,
            'n_features': len(features),
            'n_samples': len(current_data),
            'drift_summary': {
                'critical': n_critical,
                'warning': n_warning,
                'stable': n_stable,
            },
            'overall_status': 'critical' if n_critical > 0 else ('warning' if n_warning > 0 else 'stable'),
            'drift_results': drift_results,
        }
        
        # Save to history
        self.drift_history.append(report)
        self._save_drift_report(report)
        
        if n_critical > 0 or n_warning > 0:
            self.logger.warning(
                "Drift detected",
                critical=n_critical,
                warning=n_warning,
            )
        
        return report
    
    def _save_drift_report(self, report: Dict[str, Any]) -> None:
        """Save drift report to disk."""
        report_file = self.storage_path / 'drift_history.jsonl'
        with open(report_file, 'a') as f:
            f.write(json.dumps(report) + '\n')
    
    def get_drift_summary(self) -> pd.DataFrame:
        """
        Get summary of drift over time.
        
        Returns:
            DataFrame with drift history
        """
        if not self.drift_history:
            return pd.DataFrame()
        
        summaries = []
        for report in self.drift_history:
            summaries.append({
                'timestamp': report['timestamp'],
                'n_critical': report['drift_summary']['critical'],
                'n_warning': report['drift_summary']['warning'],
                'n_stable': report['drift_summary']['stable'],
                'overall_status': report['overall_status'],
            })
        
        return pd.DataFrame(summaries)
    
    def plot_drift_summary(
        self,
        report: Dict[str, Any],
        top_n: int = 20,
    ) -> None:
        """
        Plot drift summary.
        
        Args:
            report: Drift report
            top_n: Number of features to show
        """
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(report['drift_results'])
        df = df.sort_values('psi', ascending=False).head(top_n)
        
        # Color by drift level
        colors = {'critical': 'red', 'warning': 'orange', 'stable': 'green'}
        bar_colors = [colors[level] for level in df['drift_level']]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(df)), df['psi'], color=bar_colors)
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('PSI')
        plt.ylabel('Feature')
        plt.title('Feature Drift (PSI) - Top Features')
        plt.axvline(self.psi_threshold, color='orange', linestyle='--', label='Warning')
        plt.axvline(self.psi_threshold * 2.5, color='red', linestyle='--', label='Critical')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def export_drift_report(
        self,
        report: Dict[str, Any],
        filepath: str,
        format: str = 'html',
    ) -> None:
        """
        Export drift report.
        
        Args:
            report: Drift report
            filepath: Output file path
            format: Report format
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == 'excel':
            df = pd.DataFrame(report['drift_results'])
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Drift Results', index=False)
        
        elif format == 'html':
            df = pd.DataFrame(report['drift_results'])
            html = f"""
            <html>
            <head><title>Drift Detection Report</title></head>
            <body>
                <h1>Data Drift Monitoring Report</h1>
                <h2>Overall Status: {report['overall_status'].upper()}</h2>
                <p>Generated: {report['timestamp']}</p>
                
                <h3>Summary</h3>
                <ul>
                    <li>Critical: {report['drift_summary']['critical']}</li>
                    <li>Warning: {report['drift_summary']['warning']}</li>
                    <li>Stable: {report['drift_summary']['stable']}</li>
                </ul>
                
                <h3>Feature Drift Results</h3>
                {df.to_html(index=False)}
            </body>
            </html>
            """
            with open(filepath, 'w') as f:
                f.write(html)
        
        self.logger.info(f"Drift report exported to {filepath}")
