"""
Structured audit logging with JSON output for compliance and debugging.

All critical decisions (config changes, manual overrides, permissions) are logged
with full context for audit trails.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class AuditLogger:
    """
    Structured logger for audit trails and operational logging.

    Outputs JSON lines to both file and console with different log levels.
    All logs include timestamp, run_id, user_id, and context.

    Example:
        >>> logger = AuditLogger(log_file="audit.jsonl", run_id="run_123")
        >>> logger.info("Model trained", accuracy=0.95)
        >>> logger.warning("Manual override applied", variable="age", reason="business rule")
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        console_level: str = "INFO",
    ) -> None:
        """
        Initialize audit logger.

        Args:
            log_file: Path to log file (JSON Lines format). If None, logs to stdout only.
            run_id: Current run identifier for context
            user_id: User identifier for audit trail
            console_level: Minimum level for console output (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_file = Path(log_file) if log_file else None
        self.run_id = run_id
        self.user_id = user_id
        self.console_level = console_level

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                self._get_log_level(console_level)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger()

        # Ensure log file directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_log_level(self, level: str) -> int:
        """Convert string level to logging constant."""
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        return levels.get(level.upper(), 20)

    def _add_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Add run_id and user_id context to log entry."""
        context = kwargs.copy()
        if self.run_id:
            context["run_id"] = self.run_id
        if self.user_id:
            context["user_id"] = self.user_id
        return context

    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Append log entry to file as JSON line."""
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug-level message.

        Args:
            message: Human-readable message
            **kwargs: Additional context fields
        """
        context = self._add_context(**kwargs)
        self.logger.debug(message, **context)
        if self.log_file:
            self._write_to_file(
                {"level": "DEBUG", "message": message, "timestamp": datetime.now().isoformat(), **context}
            )

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info-level message.

        Args:
            message: Human-readable message
            **kwargs: Additional context fields

        Example:
            >>> logger.info("Config loaded", path="config.yml", num_features=15)
        """
        context = self._add_context(**kwargs)
        self.logger.info(message, **context)
        if self.log_file:
            self._write_to_file(
                {"level": "INFO", "message": message, "timestamp": datetime.now().isoformat(), **context}
            )

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning-level message.

        Args:
            message: Human-readable message
            **kwargs: Additional context fields

        Example:
            >>> logger.warning("Missing values detected", variable="income", pct_missing=0.15)
        """
        context = self._add_context(**kwargs)
        self.logger.warning(message, **context)
        if self.log_file:
            self._write_to_file(
                {"level": "WARNING", "message": message, "timestamp": datetime.now().isoformat(), **context}
            )

    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """
        Log error-level message with optional exception.

        Args:
            message: Human-readable message
            error: Exception instance for traceback
            **kwargs: Additional context fields

        Example:
            >>> try:
            ...     raise ValueError("Invalid config")
            ... except Exception as e:
            ...     logger.error("Config validation failed", error=e, path="config.yml")
        """
        context = self._add_context(**kwargs)
        if error:
            context["error_type"] = type(error).__name__
            context["error_message"] = str(error)
        self.logger.error(message, **context)
        if self.log_file:
            self._write_to_file(
                {"level": "ERROR", "message": message, "timestamp": datetime.now().isoformat(), **context}
            )

    def audit(
        self,
        action: str,
        resource_id: str,
        before: Optional[Any] = None,
        after: Optional[Any] = None,
        reason: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Log audit trail event for compliance.

        Required for all manual overrides, config changes, and permission decisions.

        Args:
            action: Action type (e.g., "manual_override", "config_change", "permission_check")
            resource_id: Identifier of affected resource
            before: State before change
            after: State after change
            reason: Human-readable justification
            **kwargs: Additional context

        Example:
            >>> logger.audit(
            ...     action="manual_override",
            ...     resource_id="age_binning",
            ...     before="auto_bins",
            ...     after="custom_bins",
            ...     reason="Business rule requires specific age groups"
            ... )
        """
        context = self._add_context(
            action=action,
            resource_id=resource_id,
            before=before,
            after=after,
            reason=reason,
            **kwargs
        )
        self.logger.info(f"AUDIT: {action}", **context)
        if self.log_file:
            self._write_to_file(
                {
                    "level": "AUDIT",
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    **context
                }
            )


# Global logger instance
_global_logger: Optional[AuditLogger] = None


def get_audit_logger(
    log_file: Optional[str] = None,
    run_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AuditLogger:
    """
    Get or create global audit logger instance.

    Args:
        log_file: Path to log file (only used on first call)
        run_id: Current run ID
        user_id: Current user ID

    Returns:
        Global AuditLogger instance

    Example:
        >>> logger = get_audit_logger(log_file="logs/audit.jsonl", run_id="run_123")
        >>> logger.info("Starting EDA")
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = AuditLogger(log_file=log_file, run_id=run_id, user_id=user_id)
    return _global_logger
