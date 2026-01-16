"""
Reason code generation for scorecard decisions.

Generates regulatory-compliant adverse action reason codes
for credit decisions.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cr_score.core.logging import get_audit_logger


class ReasonCodeGenerator:
    """
    Generate reason codes for scorecard decisions.
    
    Provides top factors contributing to adverse decisions
    for regulatory compliance (FCRA, ECOA).
    
    Example:
        >>> generator = ReasonCodeGenerator(model, feature_names)
        >>> reasons = generator.generate_reasons(x, score=580, threshold=620)
        >>> print(reasons)  # Top 4 adverse factors
    """
    
    # Standard reason code mappings
    REASON_CODE_MAP = {
        'age': ('RC01', 'Limited credit history length'),
        'income': ('RC02', 'Income too low relative to obligations'),
        'employment_years': ('RC03', 'Limited employment history'),
        'credit_history_years': ('RC04', 'Insufficient credit history'),
        'num_credit_lines': ('RC05', 'Limited number of credit accounts'),
        'debt_to_income_ratio': ('RC06', 'High debt-to-income ratio'),
        'credit_utilization': ('RC07', 'High credit utilization'),
        'num_recent_inquiries': ('RC08', 'Too many recent credit inquiries'),
        'num_delinquent_accounts': ('RC09', 'Delinquent accounts present'),
        'loan_amount': ('RC10', 'Requested loan amount too high'),
    }
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        reason_code_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        """
        Initialize reason code generator.
        
        Args:
            model: Fitted scorecard model
            feature_names: List of feature names
            reason_code_map: Custom reason code mappings
        """
        self.model = model
        self.feature_names = feature_names
        self.reason_code_map = reason_code_map or self.REASON_CODE_MAP
        self.logger = get_audit_logger()
    
    def generate_reasons(
        self,
        x: pd.Series,
        score: float,
        threshold: float,
        num_reasons: int = 4,
        include_positive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate reason codes for a decision.
        
        Args:
            x: Feature values for application
            score: Computed credit score
            threshold: Decision threshold
            num_reasons: Number of reasons to return
            include_positive: Include positive factors if approved
        
        Returns:
            List of reason codes with descriptions
        """
        is_adverse = score < threshold
        
        if not is_adverse and not include_positive:
            return []
        
        # Get model coefficients or feature importance
        if hasattr(self.model, 'get_coefficients'):
            coefs = self.model.get_coefficients()
            feature_impacts = {}
            for _, row in coefs.iterrows():
                feat = row['feature']
                if feat in x.index:
                    # Impact = coefficient * feature_value
                    impact = row['coefficient'] * x[feat]
                    feature_impacts[feat] = impact
        else:
            # Fallback: use feature values directly
            feature_impacts = {feat: x[feat] for feat in self.feature_names if feat in x.index}
        
        # Sort by impact (most negative = most adverse)
        if is_adverse:
            # Most negative impacts
            sorted_features = sorted(
                feature_impacts.items(),
                key=lambda item: item[1]
            )
        else:
            # Most positive impacts
            sorted_features = sorted(
                feature_impacts.items(),
                key=lambda item: item[1],
                reverse=True
            )
        
        # Generate reason codes
        reasons = []
        for feat, impact in sorted_features[:num_reasons]:
            code, description = self.reason_code_map.get(
                feat,
                ('RC99', f'Factor: {feat}')
            )
            
            reasons.append({
                'code': code,
                'description': description,
                'feature': feat,
                'value': float(x[feat]),
                'impact': float(impact),
                'adverse': is_adverse,
            })
        
        self.logger.info(
            "Generated reason codes",
            score=score,
            threshold=threshold,
            is_adverse=is_adverse,
            num_reasons=len(reasons),
        )
        
        return reasons
    
    def generate_adverse_action_notice(
        self,
        application_id: str,
        applicant_name: str,
        score: float,
        threshold: float,
        x: pd.Series,
        creditor_name: str = "Your Financial Institution",
    ) -> Dict[str, Any]:
        """
        Generate complete adverse action notice.
        
        Args:
            application_id: Application ID
            applicant_name: Applicant name
            score: Credit score
            threshold: Decision threshold
            x: Feature values
            creditor_name: Name of creditor
        
        Returns:
            Complete adverse action notice
        """
        reasons = self.generate_reasons(x, score, threshold)
        
        notice = {
            'notice_type': 'Adverse Action Notice',
            'application_id': application_id,
            'applicant_name': applicant_name,
            'creditor_name': creditor_name,
            'decision': 'DECLINED',
            'score': score,
            'threshold': threshold,
            'reasons': reasons,
            'disclosure': (
                "The following factors most influenced your credit decision. "
                "These are the principal reasons for the adverse action taken."
            ),
            'rights': (
                "You have the right to obtain a free copy of your credit report "
                "from the consumer reporting agency within 60 days. "
                "You may also dispute any inaccurate information in your report."
            ),
            'regulatory_compliance': ['FCRA', 'ECOA', 'Regulation B'],
        }
        
        return notice
    
    def batch_generate_reasons(
        self,
        X: pd.DataFrame,
        scores: np.ndarray,
        threshold: float,
        num_reasons: int = 4,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate reason codes for multiple applications.
        
        Args:
            X: Feature matrix
            scores: Credit scores
            threshold: Decision threshold
            num_reasons: Number of reasons per application
        
        Returns:
            List of reason code lists
        """
        all_reasons = []
        
        for i, (idx, x) in enumerate(X.iterrows()):
            reasons = self.generate_reasons(
                x,
                scores[i],
                threshold,
                num_reasons=num_reasons
            )
            all_reasons.append(reasons)
        
        return all_reasons
    
    def export_reason_codes(
        self,
        reasons: List[Dict[str, Any]],
        format: str = "json",
    ) -> Any:
        """
        Export reason codes in specified format.
        
        Args:
            reasons: List of reason codes
            format: Export format ('json', 'dataframe', 'text')
        
        Returns:
            Exported reasons
        """
        if format == "json":
            return reasons
        elif format == "dataframe":
            return pd.DataFrame(reasons)
        elif format == "text":
            text = []
            for i, reason in enumerate(reasons, 1):
                text.append(
                    f"{i}. {reason['code']}: {reason['description']} "
                    f"(Value: {reason['value']:.2f}, Impact: {reason['impact']:.4f})"
                )
            return "\n".join(text)
        else:
            raise ValueError(f"Unknown format: {format}")
