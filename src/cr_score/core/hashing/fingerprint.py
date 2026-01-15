"""
Content hashing and fingerprinting for deterministic reproducibility.

All artifacts are tracked by SHA256 hashes. Config + data hash = run determinism.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from cr_score.core.exceptions import CR_ScoreException


def _normalize_dict(d: Dict[str, Any]) -> str:
    """
    Normalize dictionary to canonical JSON string for hashing.

    Args:
        d: Dictionary to normalize

    Returns:
        Canonical JSON string with sorted keys

    Example:
        >>> _normalize_dict({"b": 2, "a": 1})
        '{"a": 1, "b": 2}'
    """
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute deterministic hash of configuration.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hex digest

    Example:
        >>> config = {"project": {"name": "test"}}
        >>> hash_val = compute_config_hash(config)
        >>> assert len(hash_val) == 64
    """
    canonical = _normalize_dict(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of file contents.

    Args:
        file_path: Path to file
        chunk_size: Bytes to read per iteration

    Returns:
        SHA256 hex digest

    Raises:
        FileNotFoundError: If file does not exist

    Example:
        >>> hash_val = compute_file_hash("data.csv")
        >>> assert len(hash_val) == 64
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_data_hash(df: pd.DataFrame, sample_size: int = 10000) -> str:
    """
    Compute deterministic hash of DataFrame.

    Uses schema + sampled rows for efficiency on large datasets.

    Args:
        df: DataFrame to hash
        sample_size: Number of rows to sample for hash

    Returns:
        SHA256 hex digest

    Example:
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> hash_val = compute_data_hash(df)
        >>> assert len(hash_val) == 64
    """
    sha256 = hashlib.sha256()

    # Hash schema
    schema_str = json.dumps(
        {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
        },
        sort_keys=True,
    )
    sha256.update(schema_str.encode("utf-8"))

    # Hash sampled rows
    if len(df) > sample_size:
        # Deterministic sampling
        indices = list(range(0, len(df), len(df) // sample_size))[:sample_size]
        sample = df.iloc[indices]
    else:
        sample = df

    # Convert to bytes
    for row in sample.itertuples(index=False):
        row_str = str(row)
        sha256.update(row_str.encode("utf-8"))

    return sha256.hexdigest()


def generate_run_id(config_hash: str, data_hash: str, timestamp: datetime) -> str:
    """
    Generate unique run identifier.

    Format: run_{timestamp}_{config_data_hash}

    Args:
        config_hash: Configuration hash (first 8 chars used)
        data_hash: Data hash (first 8 chars used)
        timestamp: Run start timestamp

    Returns:
        Unique run ID string

    Example:
        >>> from datetime import datetime
        >>> run_id = generate_run_id("abc" * 22, "def" * 22, datetime(2026, 1, 15, 10, 30))
        >>> assert run_id.startswith("run_20260115_103000_")
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    combined_hash = f"{config_hash[:8]}{data_hash[:8]}"
    return f"run_{ts_str}_{combined_hash}"


def compute_artifact_hash(artifact_path: Union[str, Path]) -> str:
    """
    Compute hash of artifact file for integrity verification.

    Args:
        artifact_path: Path to artifact file

    Returns:
        SHA256 hex digest

    Raises:
        FileNotFoundError: If artifact does not exist

    Example:
        >>> hash_val = compute_artifact_hash("artifacts/run_123/binning.csv")
        >>> assert len(hash_val) == 64
    """
    return compute_file_hash(artifact_path)


def verify_artifact_integrity(
    artifact_path: Union[str, Path], expected_hash: str
) -> bool:
    """
    Verify artifact has not been tampered with.

    Args:
        artifact_path: Path to artifact file
        expected_hash: Expected SHA256 hash

    Returns:
        True if hashes match, False otherwise

    Example:
        >>> is_valid = verify_artifact_integrity("artifact.csv", "abc123...")
        >>> assert is_valid
    """
    try:
        actual_hash = compute_artifact_hash(artifact_path)
        return actual_hash == expected_hash
    except Exception:
        return False
