"""
Spark execution metrics collection.

Tracks job, stage, and task-level metrics for performance monitoring.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from pyspark.sql import SparkSession
from pyspark import SparkContext

from cr_score.core.logging import get_audit_logger


class SparkExecutionMetrics:
    """
    Collect and track Spark execution metrics.
    
    Provides job, stage, and task-level metrics for performance analysis.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize metrics collector.
        
        Args:
            spark: Spark session (auto-detected if None)
        """
        self.spark = spark or SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("No active Spark session found. Provide spark parameter or create a session.")
        
        self.sc: SparkContext = self.spark.sparkContext
        self.logger = get_audit_logger()
        self._metrics_history: List[Dict[str, Any]] = []
    
    def collect_job_metrics(self, job_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect metrics for a specific job or latest job.
        
        Args:
            job_id: Job ID (None = latest job)
            
        Returns:
            Dictionary with job metrics
        """
        try:
            status_tracker = self.sc.statusTracker()
            
            if job_id is None:
                # Get latest job
                job_ids = status_tracker.getJobIdsForGroup(None)
                if not job_ids:
                    return {"error": "No jobs found"}
                job_id = max(job_ids)
            
            job_info = status_tracker.getJobInfo(job_id)
            if not job_info:
                return {"error": f"Job {job_id} not found"}
            
            # Collect stage metrics
            stage_metrics = []
            for stage_id in job_info.stageIds:
                stage_info = status_tracker.getStageInfo(stage_id)
                if stage_info:
                    stage_metrics.append(self._extract_stage_metrics(stage_info))
            
            return {
                "job_id": job_id,
                "status": job_info.status.name if hasattr(job_info.status, 'name') else str(job_info.status),
                "submission_time": job_info.submissionTime,
                "completion_time": job_info.completionTime if job_info.completionTime else None,
                "num_tasks": job_info.numTasks,
                "num_completed_tasks": job_info.numCompletedTasks,
                "num_active_tasks": job_info.numActiveTasks,
                "num_failed_tasks": job_info.numFailedTasks,
                "num_skipped_tasks": job_info.numSkippedTasks,
                "stages": stage_metrics,
                "collected_at": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error collecting job metrics: {e}")
            return {"error": str(e)}
    
    def _extract_stage_metrics(self, stage_info) -> Dict[str, Any]:
        """Extract metrics from stage info."""
        try:
            return {
                "stage_id": stage_info.stageId,
                "name": stage_info.name,
                "num_tasks": stage_info.numTasks,
                "num_active_tasks": stage_info.numActiveTasks,
                "num_completed_tasks": stage_info.numCompletedTasks,
                "num_failed_tasks": stage_info.numFailedTasks,
                "submission_time": stage_info.submissionTime,
                "completion_time": stage_info.completionTime if stage_info.completionTime else None,
            }
        except Exception as e:
            self.logger.warning(f"Error extracting stage metrics: {e}")
            return {"error": str(e)}
    
    def get_executor_metrics(self) -> List[Dict[str, Any]]:
        """
        Get metrics for all executors.
        
        Returns:
            List of executor metrics dictionaries
        """
        try:
            status_tracker = self.sc.statusTracker()
            executor_infos = status_tracker.getExecutorInfos()
            
            metrics = []
            for executor_info in executor_infos:
                metrics.append({
                    "executor_id": executor_info.executorId,
                    "host": executor_info.executorHost,
                    "total_cores": executor_info.totalCores,
                    "max_tasks": executor_info.maxTasks,
                    "active_tasks": executor_info.activeTasks,
                    "completed_tasks": executor_info.completedTasks,
                    "failed_tasks": executor_info.failedTasks,
                    "total_duration": executor_info.totalDuration,
                    "total_gc_time": executor_info.totalGCTime,
                    "total_input_bytes": executor_info.totalInputBytes,
                    "total_shuffle_read": executor_info.totalShuffleRead,
                    "total_shuffle_write": executor_info.totalShuffleWrite,
                    "max_memory": executor_info.maxMemory,
                })
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error collecting executor metrics: {e}")
            return []
    
    def get_spark_config(self) -> Dict[str, str]:
        """
        Get current Spark configuration.
        
        Returns:
            Dictionary with Spark configuration
        """
        config = {}
        for key, value in self.spark.conf.getAll().items():
            config[key] = value
        return config
    
    def get_rdd_info(self, rdd_id: int) -> Dict[str, Any]:
        """
        Get information about an RDD.
        
        Args:
            rdd_id: RDD ID
            
        Returns:
            Dictionary with RDD information
        """
        try:
            status_tracker = self.sc.statusTracker()
            rdd_info = status_tracker.getRDDInfo(rdd_id)
            
            if not rdd_info:
                return {"error": f"RDD {rdd_id} not found"}
            
            return {
                "rdd_id": rdd_info.id,
                "name": rdd_info.name,
                "num_partitions": rdd_info.numPartitions,
                "num_cached_partitions": rdd_info.numCachedPartitions,
                "memory_size": rdd_info.memorySize,
                "disk_size": rdd_info.diskSize,
            }
        except Exception as e:
            self.logger.error(f"Error getting RDD info: {e}")
            return {"error": str(e)}
    
    def track_execution(
        self,
        operation_name: str,
        func,
        *args,
        **kwargs,
    ) -> tuple:
        """
        Track execution of a function and collect metrics.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, metrics)
        """
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Collect job metrics
            job_metrics = self.collect_job_metrics()
            
            metrics = {
                "operation": operation_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "job_metrics": job_metrics,
            }
            
            self._metrics_history.append(metrics)
            self.logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
            
            return result, metrics
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = {
                "operation": operation_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e),
            }
            
            self._metrics_history.append(metrics)
            self.logger.error(f"Operation '{operation_name}' failed after {duration:.2f}s: {e}")
            raise
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all tracked operations.
        
        Returns:
            List of metrics dictionaries
        """
        return self._metrics_history.copy()
    
    def export_metrics(self, path: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            path: Path to export file
        """
        export_data = {
            "spark_config": self.get_spark_config(),
            "executor_metrics": self.get_executor_metrics(),
            "execution_history": self._metrics_history,
            "exported_at": datetime.now().isoformat(),
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics to {path}")
