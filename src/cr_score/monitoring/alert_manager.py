"""
Alert management for scorecard monitoring.

Handles alert generation, routing, and notification.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cr_score.core.logging import get_audit_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """
    Manage monitoring alerts.
    
    Generates, stores, and routes alerts to appropriate channels.
    
    Example:
        >>> manager = AlertManager()
        >>> manager.create_alert(
        ...     title="Model Performance Degradation",
        ...     severity=AlertSeverity.CRITICAL,
        ...     details={"auc_drop": 0.08}
        ... )
        >>> manager.send_alerts()
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        notification_handlers: Optional[List[Callable]] = None,
    ) -> None:
        """
        Initialize alert manager.
        
        Args:
            storage_path: Path to store alerts
            notification_handlers: List of notification functions
        """
        self.storage_path = Path(storage_path) if storage_path else Path('./monitoring_data/alerts')
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.notification_handlers = notification_handlers or []
        self.logger = get_audit_logger()
        
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
    
    def create_alert(
        self,
        title: str,
        severity: AlertSeverity,
        details: Dict[str, Any],
        source: str = "scorecard_monitor",
    ) -> Dict[str, Any]:
        """
        Create new alert.
        
        Args:
            title: Alert title
            severity: Alert severity
            details: Alert details
            source: Alert source
        
        Returns:
            Created alert
        """
        alert = {
            'alert_id': self._generate_alert_id(),
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'severity': severity.value,
            'details': details,
            'source': source,
            'status': 'active',
            'acknowledged': False,
        }
        
        self.active_alerts.append(alert)
        self._save_alert(alert)
        
        self.logger.info(
            f"Alert created: {title}",
            severity=severity.value,
            alert_id=alert['alert_id'],
        )
        
        # Send notifications for critical alerts
        if severity == AlertSeverity.CRITICAL:
            self._send_notification(alert)
        
        return alert
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import hashlib
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _save_alert(self, alert: Dict[str, Any]) -> None:
        """Save alert to disk."""
        alert_file = self.storage_path / f"{alert['alert_id']}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        # Also append to history
        history_file = self.storage_path / 'alert_history.jsonl'
        with open(history_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def _send_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
        
        Returns:
            True if successful
        """
        for alert in self.active_alerts:
            if alert['alert_id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                self._save_alert(alert)
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution_notes: Resolution notes
        
        Returns:
            True if successful
        """
        for i, alert in enumerate(self.active_alerts):
            if alert['alert_id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_notes'] = resolution_notes
                
                # Move to history
                self.alert_history.append(alert)
                self.active_alerts.pop(i)
                
                self._save_alert(alert)
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts.
        
        Args:
            severity: Filter by severity
        
        Returns:
            List of active alerts
        """
        if severity:
            return [a for a in self.active_alerts if a['severity'] == severity.value]
        return self.active_alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alerts.
        
        Returns:
            Alert summary
        """
        summary = {
            'total_active': len(self.active_alerts),
            'by_severity': {
                'critical': len([a for a in self.active_alerts if a['severity'] == 'critical']),
                'warning': len([a for a in self.active_alerts if a['severity'] == 'warning']),
                'info': len([a for a in self.active_alerts if a['severity'] == 'info']),
            },
            'unacknowledged': len([a for a in self.active_alerts if not a['acknowledged']]),
            'total_history': len(self.alert_history),
        }
        
        return summary
    
    def export_alerts(self, filepath: str, format: str = 'json') -> None:
        """
        Export alerts to file.
        
        Args:
            filepath: Output file path
            format: Export format
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump({
                    'active': self.active_alerts,
                    'history': self.alert_history,
                    'summary': self.get_alert_summary(),
                }, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(self.active_alerts + self.alert_history)
            df.to_csv(filepath, index=False)
        
        self.logger.info(f"Alerts exported to {filepath}")


# Example notification handlers
def email_notification_handler(alert: Dict[str, Any]) -> None:
    """
    Example email notification handler.
    
    In production, implement actual email sending logic.
    """
    print(f"[EMAIL] Alert: {alert['title']} - Severity: {alert['severity']}")


def slack_notification_handler(alert: Dict[str, Any]) -> None:
    """
    Example Slack notification handler.
    
    In production, implement actual Slack webhook.
    """
    print(f"[SLACK] Alert: {alert['title']} - Severity: {alert['severity']}")


def pagerduty_notification_handler(alert: Dict[str, Any]) -> None:
    """
    Example PagerDuty notification handler.
    
    In production, implement actual PagerDuty API call.
    """
    print(f"[PAGERDUTY] Alert: {alert['title']} - Severity: {alert['severity']}")
