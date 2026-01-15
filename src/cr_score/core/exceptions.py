"""
CR_Score exception hierarchy.

All custom exceptions inherit from CR_ScoreException for consistent error handling.
"""

from typing import Any, Dict, Optional


class CR_ScoreException(Exception):
    """Base exception for all CR_Score errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error description
            details: Additional context for debugging and logging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """String representation including details."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigValidationError(CR_ScoreException):
    """Configuration schema validation failed."""

    pass


class DataValidationError(CR_ScoreException):
    """Data quality or schema validation failed."""

    pass


class ArtifactNotFoundError(CR_ScoreException):
    """Requested artifact does not exist."""

    pass


class ArtifactIntegrityError(CR_ScoreException):
    """Artifact hash mismatch or corruption detected."""

    pass


class RunNotFoundError(CR_ScoreException):
    """Run ID not found in registry."""

    pass


class PermissionDeniedError(CR_ScoreException):
    """User lacks required permission for operation."""

    pass


class CompressionError(CR_ScoreException):
    """Data compression verification failed."""

    pass


class SparkSessionError(CR_ScoreException):
    """Spark session creation or operation failed."""

    pass


class BinningError(CR_ScoreException):
    """Binning quality gate or constraint violation."""

    pass


class ModelingError(CR_ScoreException):
    """Model training or validation error."""

    pass


class ReproducibilityError(CR_ScoreException):
    """Determinism verification failed."""

    pass
