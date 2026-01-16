"""
Report exporter for multiple output formats.

Exports scorecard results to JSON, CSV, Excel, and other formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ReportExporter:
    """
    Export scorecard reports in multiple formats.
    
    Supports JSON, CSV, Excel, and Markdown for machine-readable outputs.
    
    Example:
        >>> exporter = ReportExporter()
        >>> exporter.export_comprehensive_report(
        ...     model=model,
        ...     metrics=metrics,
        ...     output_dir='reports/',
        ...     formats=['json', 'csv', 'excel']
        ... )
    """
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Convert numpy/pandas objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: ReportExporter._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ReportExporter._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def export_metrics_json(
        self,
        metrics: Dict[str, Any],
        filepath: Union[str, Path],
        pretty: bool = True,
    ) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            filepath: Output file path
            pretty: Whether to pretty-print JSON
        
        Example:
            >>> exporter.export_metrics_json(metrics, 'metrics.json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_metrics = self._convert_to_serializable(metrics)
        
        # Add metadata
        output = {
            "timestamp": datetime.now().isoformat(),
            "cr_score_version": "1.2.0",
            "metrics": serializable_metrics,
        }
        
        # Write JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(output, f, indent=2, ensure_ascii=False)
            else:
                json.dump(output, f, ensure_ascii=False)
    
    def export_metrics_csv(
        self,
        metrics: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """
        Export metrics to multiple CSV files.
        
        Args:
            metrics: Metrics dictionary
            output_dir: Output directory
        
        Returns:
            List of created file paths
        
        Example:
            >>> files = exporter.export_metrics_csv(metrics, 'reports/')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # Classification metrics
        if "classification" in metrics:
            clf_df = pd.DataFrame([metrics["classification"]])
            filepath = output_dir / "classification_metrics.csv"
            clf_df.to_csv(filepath, index=False)
            created_files.append(filepath)
        
        # Ranking metrics
        if "ranking" in metrics:
            rank_df = pd.DataFrame([metrics["ranking"]])
            filepath = output_dir / "ranking_metrics.csv"
            rank_df.to_csv(filepath, index=False)
            created_files.append(filepath)
        
        # Calibration metrics
        if "calibration" in metrics:
            calib_df = pd.DataFrame([metrics["calibration"]])
            filepath = output_dir / "calibration_metrics.csv"
            calib_df.to_csv(filepath, index=False)
            created_files.append(filepath)
        
        # Stability metrics
        if "stability" in metrics:
            stab_df = pd.DataFrame([metrics["stability"]])
            filepath = output_dir / "stability_metrics.csv"
            stab_df.to_csv(filepath, index=False)
            created_files.append(filepath)
        
        return created_files
    
    def export_to_excel(
        self,
        metrics: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None,
        lift_curve: Optional[pd.DataFrame] = None,
        gains_curve: Optional[pd.DataFrame] = None,
        psi_breakdown: Optional[pd.DataFrame] = None,
        filepath: Union[str, Path] = "scorecard_report.xlsx",
    ) -> None:
        """
        Export comprehensive report to Excel with multiple sheets.
        
        Args:
            metrics: Metrics dictionary
            feature_importance: Feature importance DataFrame
            lift_curve: Lift curve DataFrame
            gains_curve: Gains curve DataFrame
            psi_breakdown: PSI breakdown DataFrame
            filepath: Output file path
        
        Example:
            >>> exporter.export_to_excel(
            ...     metrics=metrics,
            ...     feature_importance=importance_df,
            ...     filepath='report.xlsx'
            ... )
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            
            if "classification" in metrics:
                for key, value in metrics["classification"].items():
                    if not isinstance(value, dict):
                        summary_data.append({"Category": "Classification", "Metric": key, "Value": value})
            
            if "ranking" in metrics:
                for key, value in metrics["ranking"].items():
                    if not isinstance(value, dict):
                        summary_data.append({"Category": "Ranking", "Metric": key, "Value": value})
            
            if "calibration" in metrics:
                for key, value in metrics["calibration"].items():
                    if not isinstance(value, dict):
                        summary_data.append({"Category": "Calibration", "Metric": key, "Value": value})
            
            if "stability" in metrics:
                for key, value in metrics["stability"].items():
                    if not isinstance(value, dict):
                        summary_data.append({"Category": "Stability", "Metric": key, "Value": value})
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Feature importance
            if feature_importance is not None:
                feature_importance.to_excel(writer, sheet_name='Feature Importance', index=False)
            
            # Lift curve
            if lift_curve is not None:
                lift_curve.to_excel(writer, sheet_name='Lift Curve', index=False)
            
            # Gains curve
            if gains_curve is not None:
                gains_curve.to_excel(writer, sheet_name='Gains Curve', index=False)
            
            # PSI breakdown
            if psi_breakdown is not None:
                psi_breakdown.to_excel(writer, sheet_name='PSI Breakdown', index=False)
            
            # Interpretations
            if "interpretations" in metrics:
                interp_df = pd.DataFrame([
                    {"Metric": key, "Interpretation": value}
                    for key, value in metrics["interpretations"].items()
                ])
                interp_df.to_excel(writer, sheet_name='Interpretations', index=False)
    
    def export_to_markdown(
        self,
        metrics: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None,
        filepath: Union[str, Path] = "scorecard_report.md",
    ) -> None:
        """
        Export report to Markdown format.
        
        Args:
            metrics: Metrics dictionary
            feature_importance: Feature importance DataFrame
            filepath: Output file path
        
        Example:
            >>> exporter.export_to_markdown(metrics, importance_df, 'report.md')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        lines.append("# Scorecard Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Classification Metrics
        if "classification" in metrics:
            lines.append("## Classification Metrics\n")
            clf = metrics["classification"]
            lines.append(f"- **Accuracy**: {clf.get('accuracy', 0):.4f}")
            lines.append(f"- **Precision**: {clf.get('precision', 0):.4f}")
            lines.append(f"- **Recall**: {clf.get('recall', 0):.4f}")
            lines.append(f"- **F1 Score**: {clf.get('f1_score', 0):.4f}")
            lines.append(f"- **MCC**: {clf.get('mcc', 0):.4f}\n")
        
        # Ranking Metrics
        if "ranking" in metrics:
            lines.append("## Ranking Metrics\n")
            rank = metrics["ranking"]
            lines.append(f"- **AUC**: {rank.get('auc', 0):.4f}")
            lines.append(f"- **Gini**: {rank.get('gini', 0):.4f}")
            lines.append(f"- **KS Statistic**: {rank.get('ks_statistic', 0):.4f}\n")
        
        # Calibration Metrics
        if "calibration" in metrics:
            lines.append("## Calibration Metrics\n")
            calib = metrics["calibration"]
            lines.append(f"- **Brier Score**: {calib.get('brier_score', 0):.4f}")
            lines.append(f"- **Log Loss**: {calib.get('log_loss', 0):.4f}")
            lines.append(f"- **ECE**: {calib.get('ece', 0):.4f}\n")
        
        # Interpretations
        if "interpretations" in metrics:
            lines.append("## Interpretations\n")
            for metric, interpretation in metrics["interpretations"].items():
                lines.append(f"- **{metric}**: {interpretation}")
            lines.append("")
        
        # Feature Importance
        if feature_importance is not None:
            lines.append("## Feature Importance\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|------|---------|------------|")
            for idx, row in feature_importance.head(20).iterrows():
                feature = row.get('feature', row.get('Feature', 'Unknown'))
                importance = row.get('importance', row.get('Importance', row.get('coefficient', 0)))
                lines.append(f"| {idx + 1} | {feature} | {importance:.4f} |")
            lines.append("")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def export_comprehensive_report(
        self,
        model: Any,
        metrics: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: Union[str, Path] = "reports/",
        formats: List[str] = ['json', 'csv', 'excel', 'markdown'],
        include_curves: bool = True,
    ) -> Dict[str, List[Path]]:
        """
        Export comprehensive report in multiple formats.
        
        Args:
            model: Trained model
            metrics: Metrics dictionary
            X_test: Test features
            y_test: Test labels
            output_dir: Output directory
            formats: List of formats to export ('json', 'csv', 'excel', 'markdown')
            include_curves: Whether to include lift/gains curves
        
        Returns:
            Dictionary mapping formats to created file paths
        
        Example:
            >>> files = exporter.export_comprehensive_report(
            ...     model=model,
            ...     metrics=metrics,
            ...     X_test=X_test,
            ...     y_test=y_test,
            ...     output_dir='reports/',
            ...     formats=['json', 'excel']
            ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        # Get additional data
        y_proba = model.predict_proba(X_test)[:, 1]
        feature_importance = model.get_feature_importance()
        
        lift_curve = None
        gains_curve = None
        if include_curves:
            lift_curve = model.get_lift_curve(y_test, y_proba, n_bins=10)
            gains_curve = model.get_gains_curve(y_test, y_proba, n_bins=10)
        
        psi_breakdown = None
        if "stability" in metrics and "breakdown" in metrics["stability"]:
            psi_breakdown = metrics["stability"]["breakdown"]
        
        # Export in requested formats
        if 'json' in formats:
            json_path = output_dir / "metrics.json"
            self.export_metrics_json(metrics, json_path)
            created_files['json'] = [json_path]
        
        if 'csv' in formats:
            csv_files = self.export_metrics_csv(metrics, output_dir)
            created_files['csv'] = csv_files
        
        if 'excel' in formats:
            excel_path = output_dir / "scorecard_report.xlsx"
            self.export_to_excel(
                metrics=metrics,
                feature_importance=feature_importance,
                lift_curve=lift_curve,
                gains_curve=gains_curve,
                psi_breakdown=psi_breakdown,
                filepath=excel_path,
            )
            created_files['excel'] = [excel_path]
        
        if 'markdown' in formats:
            md_path = output_dir / "scorecard_report.md"
            self.export_to_markdown(metrics, feature_importance, md_path)
            created_files['markdown'] = [md_path]
        
        return created_files
