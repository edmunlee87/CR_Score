"""
Spark performance profiling utilities.

Provides profiling tools for analyzing Spark job performance and bottlenecks.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import time

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession

from cr_score.spark.metrics.execution_metrics import SparkExecutionMetrics
from cr_score.core.logging import get_audit_logger


class PerformanceProfiler:
    """
    Performance profiler for Spark operations.
    
    Analyzes execution time, identifies bottlenecks, and provides optimization suggestions.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize profiler.
        
        Args:
            spark: Spark session (auto-detected if None)
        """
        self.spark = spark or SparkSession.getActiveSession()
        if self.spark is None:
            raise ValueError("No active Spark session found. Provide spark parameter or create a session.")
        
        self.metrics = SparkExecutionMetrics(spark)
        self.logger = get_audit_logger()
        self._profile_history: List[Dict[str, Any]] = []
    
    def profile_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Profile a Spark operation.
        
        Args:
            operation_name: Name of the operation
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with profiling results
        """
        start_time = time.time()
        start_datetime = datetime.now()
        
        # Collect initial metrics
        initial_executors = len(self.metrics.get_executor_metrics())
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_datetime = datetime.now()
            duration = end_time - start_time
            
            # Collect final metrics
            final_executors = len(self.metrics.get_executor_metrics())
            job_metrics = self.metrics.collect_job_metrics()
            
            profile = {
                "operation": operation_name,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "initial_executors": initial_executors,
                "final_executors": final_executors,
                "job_metrics": job_metrics,
                "result_type": type(result).__name__,
            }
            
            # If result is DataFrame, add DataFrame metrics
            if isinstance(result, SparkDataFrame):
                profile["dataframe_metrics"] = self._profile_dataframe(result)
            
            self._profile_history.append(profile)
            self.logger.info(
                f"Profiled '{operation_name}': {duration:.2f}s "
                f"({profile.get('dataframe_metrics', {}).get('num_partitions', 'N/A')} partitions)"
            )
            
            return profile
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            profile = {
                "operation": operation_name,
                "start_time": start_datetime.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": duration,
                "status": "failed",
                "error": str(e),
            }
            
            self._profile_history.append(profile)
            self.logger.error(f"Profiling '{operation_name}' failed: {e}")
            raise
    
    def _profile_dataframe(self, df: SparkDataFrame) -> Dict[str, Any]:
        """
        Profile a DataFrame.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dictionary with DataFrame metrics
        """
        try:
            num_partitions = df.rdd.getNumPartitions()
            
            # Get approximate row count (sampling)
            try:
                sample_count = df.limit(1000).count()
                # Rough estimate (not accurate, but fast)
                estimated_rows = sample_count * num_partitions if sample_count == 1000 else sample_count
            except Exception:
                estimated_rows = None
            
            return {
                "num_partitions": num_partitions,
                "estimated_rows": estimated_rows,
                "num_columns": len(df.columns),
                "columns": df.columns,
            }
        except Exception as e:
            self.logger.warning(f"Error profiling DataFrame: {e}")
            return {"error": str(e)}
    
    def compare_operations(
        self,
        operations: Dict[str, Callable],
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance of multiple operations.
        
        Args:
            operations: Dictionary of {name: function} to compare
            *args: Arguments to pass to all functions
            **kwargs: Keyword arguments to pass to all functions
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for name, func in operations.items():
            try:
                profile = self.profile_operation(name, func, *args, **kwargs)
                results[name] = profile
            except Exception as e:
                results[name] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        # Add comparison summary
        successful = {
            k: v for k, v in results.items()
            if v.get("status") == "success"
        }
        
        if successful:
            fastest = min(successful.items(), key=lambda x: x[1]["duration_seconds"])
            slowest = max(successful.items(), key=lambda x: x[1]["duration_seconds"])
            
            results["_comparison"] = {
                "fastest": fastest[0],
                "fastest_duration": fastest[1]["duration_seconds"],
                "slowest": slowest[0],
                "slowest_duration": slowest[1]["duration_seconds"],
                "speedup": slowest[1]["duration_seconds"] / fastest[1]["duration_seconds"],
            }
        
        return results
    
    def analyze_bottlenecks(self, profile: Dict[str, Any]) -> List[str]:
        """
        Analyze profile and identify potential bottlenecks.
        
        Args:
            profile: Profile dictionary from profile_operation
            
        Returns:
            List of bottleneck suggestions
        """
        suggestions = []
        
        duration = profile.get("duration_seconds", 0)
        job_metrics = profile.get("job_metrics", {})
        df_metrics = profile.get("dataframe_metrics", {})
        
        # Check for long duration
        if duration > 300:  # 5 minutes
            suggestions.append("Operation took >5 minutes - consider optimizing or caching intermediate results")
        
        # Check for many partitions
        num_partitions = df_metrics.get("num_partitions", 0)
        if num_partitions > 1000:
            suggestions.append(f"High partition count ({num_partitions}) - consider coalescing")
        elif num_partitions < 10:
            suggestions.append(f"Low partition count ({num_partitions}) - may benefit from repartitioning")
        
        # Check for failed tasks
        num_failed = job_metrics.get("num_failed_tasks", 0)
        if num_failed > 0:
            suggestions.append(f"Failed tasks detected ({num_failed}) - check logs for errors")
        
        # Check for many stages
        stages = job_metrics.get("stages", [])
        if len(stages) > 20:
            suggestions.append(f"Many stages ({len(stages)}) - consider checkpointing to break DAG")
        
        return suggestions
    
    def get_profile_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all profiled operations.
        
        Returns:
            List of profile dictionaries
        """
        return self._profile_history.copy()
    
    def generate_report(self) -> str:
        """
        Generate a performance report.
        
        Returns:
            Formatted report string
        """
        if not self._profile_history:
            return "No profiling data available."
        
        report_lines = ["=" * 80]
        report_lines.append("Spark Performance Profiling Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        successful = [p for p in self._profile_history if p.get("status") == "success"]
        if successful:
            total_time = sum(p["duration_seconds"] for p in successful)
            avg_time = total_time / len(successful)
            min_time = min(p["duration_seconds"] for p in successful)
            max_time = max(p["duration_seconds"] for p in successful)
            
            report_lines.append("Summary Statistics:")
            report_lines.append(f"  Total operations: {len(self._profile_history)}")
            report_lines.append(f"  Successful: {len(successful)}")
            report_lines.append(f"  Failed: {len(self._profile_history) - len(successful)}")
            report_lines.append(f"  Total time: {total_time:.2f}s")
            report_lines.append(f"  Average time: {avg_time:.2f}s")
            report_lines.append(f"  Min time: {min_time:.2f}s")
            report_lines.append(f"  Max time: {max_time:.2f}s")
            report_lines.append("")
        
        # Operation details
        report_lines.append("Operation Details:")
        for i, profile in enumerate(self._profile_history, 1):
            report_lines.append(f"\n{i}. {profile['operation']}")
            report_lines.append(f"   Duration: {profile['duration_seconds']:.2f}s")
            report_lines.append(f"   Status: {profile['status']}")
            
            if profile.get("status") == "success":
                df_metrics = profile.get("dataframe_metrics", {})
                if df_metrics:
                    report_lines.append(f"   Partitions: {df_metrics.get('num_partitions', 'N/A')}")
                    report_lines.append(f"   Columns: {df_metrics.get('num_columns', 'N/A')}")
                
                # Bottleneck analysis
                bottlenecks = self.analyze_bottlenecks(profile)
                if bottlenecks:
                    report_lines.append("   Suggestions:")
                    for suggestion in bottlenecks:
                        report_lines.append(f"     - {suggestion}")
            else:
                report_lines.append(f"   Error: {profile.get('error', 'Unknown')}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
