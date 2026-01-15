"""
Spark session factory with config-driven setup.

Manages Spark sessions for local and cluster execution with optimized defaults.
"""

from typing import Any, Dict, Optional

from pyspark import SparkConf
from pyspark.sql import SparkSession

from cr_score.core.config.schema import Config, ExecutionEngine, SparkConfig
from cr_score.core.exceptions import SparkSessionError
from cr_score.core.logging import get_audit_logger


class SparkSessionFactory:
    """
    Factory for creating and managing Spark sessions.

    Configures Spark based on execution engine and optimization settings.

    Example:
        >>> from cr_score.core.config import load_config
        >>> config = load_config("config.yml")
        >>> factory = SparkSessionFactory(config)
        >>> spark = factory.get_or_create()
        >>> df = spark.read.parquet("data.parquet")
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize Spark session factory.

        Args:
            config: CR_Score configuration with Spark settings
        """
        self.config = config
        self.logger = get_audit_logger()
        self._spark: Optional[SparkSession] = None

    def get_or_create(self) -> SparkSession:
        """
        Get existing Spark session or create new one.

        Returns:
            SparkSession instance

        Raises:
            SparkSessionError: If Spark session creation fails

        Example:
            >>> spark = factory.get_or_create()
            >>> print(spark.version)
        """
        if self._spark is not None:
            return self._spark

        engine = self.config.execution.engine

        if engine == ExecutionEngine.PYTHON_LOCAL:
            raise SparkSessionError(
                "Spark session requested but engine is python_local",
                details={"engine": engine}
            )

        self.logger.info(f"Creating Spark session for engine: {engine}")

        try:
            spark_conf = self._build_spark_conf()
            self._spark = (
                SparkSession.builder
                .config(conf=spark_conf)
                .appName(f"CR_Score_{self.config.project.name}")
                .getOrCreate()
            )

            self.logger.info(
                "Spark session created",
                version=self._spark.version,
                app_id=self._spark.sparkContext.applicationId,
            )

            return self._spark

        except Exception as e:
            raise SparkSessionError(
                f"Failed to create Spark session: {e}",
                details={"engine": engine}
            )

    def _build_spark_conf(self) -> SparkConf:
        """
        Build Spark configuration from config.

        Returns:
            SparkConf instance with optimized settings
        """
        conf = SparkConf()

        spark_config = self.config.execution.spark
        if not spark_config:
            spark_config = SparkConfig()

        # Core settings
        conf.set("spark.sql.shuffle.partitions", str(spark_config.shuffle_partitions))
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

        # Persistence
        if spark_config.persist_level:
            conf.set("spark.storage.level", spark_config.persist_level)

        # Checkpointing
        if spark_config.checkpoint_enabled and spark_config.checkpoint_dir:
            conf.set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")

        # Skew mitigation
        if spark_config.skew_mitigation_enabled:
            conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
            conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "3")
            conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")

        # Optimization
        conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
        conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")

        # Compression
        conf.set("spark.sql.parquet.compression.codec", "snappy")
        conf.set("spark.sql.inMemoryColumnarStorage.compressed", "true")

        # Engine-specific settings
        if self.config.execution.engine == ExecutionEngine.SPARK_LOCAL:
            conf.setMaster("local[*]")
            conf.set("spark.driver.memory", "4g")
            conf.set("spark.executor.memory", "4g")
        elif self.config.execution.engine == ExecutionEngine.SPARK_CLUSTER:
            # Cluster settings from environment or defaults
            conf.set("spark.dynamicAllocation.enabled", "true")
            conf.set("spark.dynamicAllocation.minExecutors", "2")
            conf.set("spark.dynamicAllocation.maxExecutors", "50")
            conf.set("spark.executor.memory", "8g")
            conf.set("spark.executor.cores", "4")
            conf.set("spark.driver.memory", "4g")

        return conf

    def stop(self) -> None:
        """
        Stop Spark session.

        Example:
            >>> factory.stop()
        """
        if self._spark is not None:
            self.logger.info("Stopping Spark session")
            self._spark.stop()
            self._spark = None

    def set_checkpoint_dir(self, checkpoint_dir: str) -> None:
        """
        Set checkpoint directory for long DAGs.

        Args:
            checkpoint_dir: Path to checkpoint directory

        Example:
            >>> factory.set_checkpoint_dir("/tmp/spark_checkpoints")
            >>> spark = factory.get_or_create()
            >>> spark.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")
        """
        spark = self.get_or_create()
        spark.sparkContext.setCheckpointDir(checkpoint_dir)
        self.logger.info(f"Checkpoint directory set to: {checkpoint_dir}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of Spark configuration.

        Returns:
            Dictionary with key Spark settings

        Example:
            >>> summary = factory.get_config_summary()
            >>> print(summary["shuffle_partitions"])
        """
        spark_config = self.config.execution.spark
        if not spark_config:
            return {}

        return {
            "engine": self.config.execution.engine.value,
            "shuffle_partitions": spark_config.shuffle_partitions,
            "persist_level": spark_config.persist_level,
            "checkpoint_enabled": spark_config.checkpoint_enabled,
            "skew_mitigation_enabled": spark_config.skew_mitigation_enabled,
            "salting_factor": spark_config.skew_mitigation_salting_factor,
        }

    @staticmethod
    def get_session() -> Optional[SparkSession]:
        """
        Get active Spark session if exists.

        Returns:
            SparkSession or None if no active session

        Example:
            >>> spark = SparkSessionFactory.get_session()
            >>> if spark:
            ...     print("Active session found")
        """
        return SparkSession.getActiveSession()
