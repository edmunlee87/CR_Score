"""
Artifact index for tracking all generated outputs with content hashes.

Every artifact (binning table, model coefficients, reports) is cataloged
with metadata, lineage, and integrity hashes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cr_score.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError
from cr_score.core.hashing.fingerprint import compute_artifact_hash, verify_artifact_integrity


class ArtifactIndex:
    """
    Index of all artifacts produced by runs.

    Tracks artifact metadata, file paths, content hashes, schema, and lineage.

    Example:
        >>> index = ArtifactIndex("artifacts/artifact_index.json")
        >>> index.register_artifact(
        ...     artifact_id="binning_age_v1",
        ...     artifact_type="binning_table",
        ...     run_id="run_123",
        ...     file_paths=["binning_tables/age.csv"],
        ...     schema={"columns": ["bin", "count", "event_rate"]}
        ... )
        >>> artifact = index.get_artifact("binning_age_v1")
    """

    def __init__(self, index_file: str = "artifact_index.json") -> None:
        """
        Initialize artifact index.

        Args:
            index_file: Path to JSON index file

        Note:
            Index file is created if it doesn't exist.
        """
        self.index_file = Path(index_file)
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load index from file or create empty index."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save index to file."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2, default=str)

    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        run_id: str,
        file_paths: List[str],
        schema: Optional[Dict[str, Any]] = None,
        lineage: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register new artifact in index.

        Args:
            artifact_id: Unique artifact identifier
            artifact_type: Type (e.g., "binning_table", "model_coefficients", "eda_report")
            run_id: Associated run ID
            file_paths: List of file paths relative to artifact root
            schema: Schema or structure definition
            lineage: List of input artifact IDs this was derived from
            metadata: Additional metadata

        Returns:
            The artifact_id

        Raises:
            ArtifactIntegrityError: If file doesn't exist or can't be hashed

        Example:
            >>> index.register_artifact(
            ...     artifact_id="model_coef_v1",
            ...     artifact_type="model_coefficients",
            ...     run_id="run_123",
            ...     file_paths=["model/coefficients.csv"],
            ...     schema={"columns": ["variable", "coefficient", "std_error"]},
            ...     lineage=["binning_age_v1", "binning_income_v1"],
            ...     metadata={"model_type": "logistic", "n_features": 15}
            ... )
        """
        # Compute content hashes
        content_hashes = {}
        for file_path in file_paths:
            try:
                content_hashes[file_path] = compute_artifact_hash(file_path)
            except FileNotFoundError:
                raise ArtifactIntegrityError(
                    f"Cannot hash artifact file: {file_path}",
                    details={"artifact_id": artifact_id}
                )

        # Create artifact entry
        artifact_entry = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "run_id": run_id,
            "file_paths": file_paths,
            "content_hashes": content_hashes,
            "schema": schema or {},
            "lineage": lineage or [],
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }

        self.index[artifact_id] = artifact_entry
        self._save_index()
        return artifact_id

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get artifact metadata.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Artifact metadata dictionary

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist

        Example:
            >>> artifact = index.get_artifact("binning_age_v1")
            >>> print(artifact["artifact_type"])
        """
        if artifact_id not in self.index:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_id}")
        return self.index[artifact_id]

    def list_artifacts(
        self,
        run_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List artifacts with optional filtering.

        Args:
            run_id: Filter by run ID
            artifact_type: Filter by artifact type

        Returns:
            List of artifact metadata dictionaries

        Example:
            >>> artifacts = index.list_artifacts(run_id="run_123", artifact_type="binning_table")
            >>> for artifact in artifacts:
            ...     print(artifact["artifact_id"])
        """
        results = []
        for artifact in self.index.values():
            if run_id and artifact["run_id"] != run_id:
                continue
            if artifact_type and artifact["artifact_type"] != artifact_type:
                continue
            results.append(artifact)

        return sorted(results, key=lambda x: x["created_at"], reverse=True)

    def verify_artifact(self, artifact_id: str) -> bool:
        """
        Verify artifact integrity by checking content hashes.

        Args:
            artifact_id: Artifact identifier

        Returns:
            True if all files match their stored hashes

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist

        Example:
            >>> is_valid = index.verify_artifact("binning_age_v1")
            >>> assert is_valid
        """
        artifact = self.get_artifact(artifact_id)

        for file_path, expected_hash in artifact["content_hashes"].items():
            if not verify_artifact_integrity(file_path, expected_hash):
                return False

        return True

    def get_artifact_lineage(self, artifact_id: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get full lineage tree for artifact.

        Args:
            artifact_id: Artifact identifier
            depth: Maximum depth to traverse

        Returns:
            Nested dictionary showing artifact lineage tree

        Example:
            >>> lineage = index.get_artifact_lineage("model_coef_v1")
            >>> print(lineage)
        """
        if depth == 0:
            return {}

        try:
            artifact = self.get_artifact(artifact_id)
        except ArtifactNotFoundError:
            return {}

        lineage_tree = {
            "artifact_id": artifact_id,
            "artifact_type": artifact["artifact_type"],
            "run_id": artifact["run_id"],
            "created_at": artifact["created_at"],
            "parents": []
        }

        for parent_id in artifact.get("lineage", []):
            parent_lineage = self.get_artifact_lineage(parent_id, depth - 1)
            if parent_lineage:
                lineage_tree["parents"].append(parent_lineage)

        return lineage_tree

    def delete_artifact(self, artifact_id: str, delete_files: bool = False) -> None:
        """
        Remove artifact from index and optionally delete files.

        Args:
            artifact_id: Artifact identifier
            delete_files: If True, also delete artifact files from disk

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist

        Example:
            >>> index.delete_artifact("temp_artifact", delete_files=True)
        """
        artifact = self.get_artifact(artifact_id)

        if delete_files:
            for file_path in artifact["file_paths"]:
                path = Path(file_path)
                if path.exists():
                    path.unlink()

        del self.index[artifact_id]
        self._save_index()

    def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            List of artifact metadata dictionaries

        Example:
            >>> artifacts = index.get_run_artifacts("run_123")
            >>> print(f"Run produced {len(artifacts)} artifacts")
        """
        return self.list_artifacts(run_id=run_id)

    def export_index(self, output_path: str) -> None:
        """
        Export index to JSON file.

        Args:
            output_path: Output file path

        Example:
            >>> index.export_index("backup/artifact_index_backup.json")
        """
        with open(output_path, "w") as f:
            json.dump(self.index, f, indent=2, default=str)
