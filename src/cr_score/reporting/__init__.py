"""Reporting module for scorecard documentation and observability dashboards."""

from cr_score.reporting.html_report import HTMLReportGenerator
from cr_score.reporting.observability_dashboard import ObservabilityDashboard

__all__ = [
    "HTMLReportGenerator",
    "ObservabilityDashboard",
]
