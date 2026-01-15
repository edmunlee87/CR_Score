"""
Run registry for tracking execution history and metadata.

Stores run metadata in SQLite (local) or PostgreSQL (production).
Each run is uniquely identified and linked to config/data hashes.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cr_score.core.exceptions import RunNotFoundError


class RunRegistry:
    """
    Registry for tracking scorecard development runs.

    Stores metadata: run_id, config_hash, data_hash, status, timestamps,
    user_id, and execution profile.

    Example:
        >>> registry = RunRegistry("runs.db")
        >>> run_id = registry.create_run(
        ...     run_id="run_123",
        ...     config_hash="abc...",
        ...     data_hash="def...",
        ...     user_id="analyst1"
        ... )
        >>> registry.update_status(run_id, "completed")
    """

    def __init__(self, db_path: str = "cr_score_runs.db") -> None:
        """
        Initialize run registry with SQLite backend.

        Args:
            db_path: Path to SQLite database file

        Note:
            Database file and tables are created if they don't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                user_id TEXT,
                project_name TEXT,
                execution_engine TEXT,
                config_path TEXT,
                error_message TEXT,
                execution_profile TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                step_order INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_seconds REAL,
                rows_in INTEGER,
                rows_out INTEGER,
                error_message TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id)
        """)

        conn.commit()
        conn.close()

    def create_run(
        self,
        run_id: str,
        config_hash: str,
        data_hash: str,
        user_id: Optional[str] = None,
        project_name: Optional[str] = None,
        execution_engine: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> str:
        """
        Create new run entry.

        Args:
            run_id: Unique run identifier
            config_hash: Configuration hash
            data_hash: Data hash
            user_id: User who initiated run
            project_name: Project name from config
            execution_engine: Engine type (python_local, spark_local, spark_cluster)
            config_path: Path to config file

        Returns:
            The run_id

        Example:
            >>> registry.create_run(
            ...     run_id="run_20260115_123456_abc123",
            ...     config_hash="abc...",
            ...     data_hash="def...",
            ...     user_id="analyst1"
            ... )
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO runs (
                run_id, config_hash, data_hash, status, created_at, updated_at,
                user_id, project_name, execution_engine, config_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                config_hash,
                data_hash,
                "created",
                now,
                now,
                user_id,
                project_name,
                execution_engine,
                config_path,
            ),
        )

        conn.commit()
        conn.close()
        return run_id

    def update_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update run status.

        Args:
            run_id: Run identifier
            status: New status (created, running, completed, failed, cancelled)
            error_message: Error details if failed

        Raises:
            RunNotFoundError: If run_id not found

        Example:
            >>> registry.update_status("run_123", "running")
            >>> registry.update_status("run_123", "failed", error_message="Data validation error")
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status == "completed":
            cursor.execute(
                """
                UPDATE runs
                SET status = ?, updated_at = ?, completed_at = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status, now, now, error_message, run_id),
            )
        else:
            cursor.execute(
                """
                UPDATE runs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status, now, error_message, run_id),
            )

        if cursor.rowcount == 0:
            conn.close()
            raise RunNotFoundError(f"Run not found: {run_id}")

        conn.commit()
        conn.close()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get run metadata.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with run metadata

        Raises:
            RunNotFoundError: If run_id not found

        Example:
            >>> run_meta = registry.get_run("run_123")
            >>> print(run_meta["status"])
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise RunNotFoundError(f"Run not found: {run_id}")

        return dict(row)

    def list_runs(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering.

        Args:
            status: Filter by status
            user_id: Filter by user
            limit: Maximum number of runs to return

        Returns:
            List of run metadata dictionaries

        Example:
            >>> runs = registry.list_runs(status="completed", limit=10)
            >>> for run in runs:
            ...     print(run["run_id"], run["created_at"])
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM runs WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def add_step(
        self,
        run_id: str,
        step_name: str,
        step_order: int,
        status: str = "started",
        rows_in: Optional[int] = None,
        rows_out: Optional[int] = None,
    ) -> int:
        """
        Add step execution record.

        Args:
            run_id: Run identifier
            step_name: Step name (e.g., "eda", "binning", "modeling")
            step_order: Step sequence number
            status: Step status (started, completed, failed)
            rows_in: Input row count
            rows_out: Output row count

        Returns:
            Step ID

        Example:
            >>> step_id = registry.add_step("run_123", "binning", 3, rows_in=1000000)
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO run_steps (
                run_id, step_name, step_order, status, started_at, rows_in, rows_out
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, step_name, step_order, status, now, rows_in, rows_out),
        )

        step_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return step_id

    def update_step(
        self,
        step_id: int,
        status: str,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update step status.

        Args:
            step_id: Step identifier
            status: New status
            duration_seconds: Execution duration
            error_message: Error details if failed

        Example:
            >>> registry.update_step(step_id, "completed", duration_seconds=45.2)
        """
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE run_steps
            SET status = ?, completed_at = ?, duration_seconds = ?, error_message = ?
            WHERE id = ?
            """,
            (status, now, duration_seconds, error_message, step_id),
        )

        conn.commit()
        conn.close()

    def get_run_steps(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all steps for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of step metadata dictionaries ordered by step_order

        Example:
            >>> steps = registry.get_run_steps("run_123")
            >>> for step in steps:
            ...     print(f"{step['step_name']}: {step['status']}")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM run_steps WHERE run_id = ? ORDER BY step_order",
            (run_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
