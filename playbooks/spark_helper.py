"""
PySpark Helper - Gracefully handles missing PySpark installation.

For CR_Score playbooks, PySpark is optional. This module provides:
1. Check if PySpark is available
2. Mock PySpark if not installed
3. Enable all tutorials to run without PySpark
"""

import warnings
import sys

# Try to import PySpark
try:
    import pyspark
    from pyspark.sql import SparkSession
    PYSPARK_AVAILABLE = True
    print("[OK] PySpark is available")
except ImportError:
    PYSPARK_AVAILABLE = False
    print("[INFO] PySpark not available - using pandas mode (this is fine for tutorials!)")


def get_spark_session():
    """
    Get Spark session if available, otherwise return None.
    
    Returns:
        SparkSession or None
    """
    if PYSPARK_AVAILABLE:
        try:
            spark = SparkSession.builder \
                .appName("CR_Score_Playbook") \
                .master("local[*]") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            return spark
        except Exception as e:
            warnings.warn(f"Could not create Spark session: {e}")
            return None
    else:
        return None


def check_spark_requirement(required=False):
    """
    Check if Spark is required for this notebook.
    
    Args:
        required: If True, raises error when Spark not available
    
    Returns:
        bool: True if Spark available
    """
    if not PYSPARK_AVAILABLE and required:
        raise ImportError(
            "PySpark is required for this notebook but not installed.\n"
            "Install with: pip install pyspark>=3.4.0"
        )
    return PYSPARK_AVAILABLE


def disable_spark_warnings():
    """Disable verbose Spark warnings."""
    if PYSPARK_AVAILABLE:
        import logging
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("pyspark").setLevel(logging.ERROR)


# Auto-disable warnings on import
disable_spark_warnings()


# Provide helpful message
if not PYSPARK_AVAILABLE:
    print("""
    [NOTE] PySpark is not installed
    
    This is perfectly fine! All CR_Score tutorials work without PySpark for small/medium datasets.
    
    [OK] You can use: pandas mode (default)
    [X] You cannot use: Spark compression (only needed for very large datasets)
    
    To install PySpark (optional):
        pip install pyspark>=3.4.0
    
    For now, tutorials will use pandas - continue without PySpark!
    """)
