"""Path handling utilities."""

from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object

    Example:
        >>> ensure_dir("artifacts/run_123")
        PosixPath('artifacts/run_123')
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def resolve_path(path: Union[str, Path], base: Union[str, Path] = ".") -> Path:
    """
    Resolve path relative to base directory.

    Args:
        path: File or directory path
        base: Base directory (default: current directory)

    Returns:
        Absolute Path object

    Example:
        >>> resolve_path("config.yml", base="/project")
        PosixPath('/project/config.yml')
    """
    return (Path(base) / path).resolve()
