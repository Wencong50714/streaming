"""Metrics collection and statistics for video frame processing."""

from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod


class Metric(ABC):
    """Base class for metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass

    @property
    @abstractmethod
    def default_value(self) -> Any:
        """Return the default value for this metric."""
        pass

    def record(self, value: Any) -> None:
        """Record a value for this metric. Override if needed."""
        pass

    def summary(self) -> Optional[str]:
        """Return a summary string for this metric. Return None to skip."""
        return None


class ListMetric(Metric):
    """Metric that collects values into a list."""

    def __init__(self, name: str, display_format: str = "{}"):
        self._name = name
        self._values: List[Any] = []
        self._display_format = display_format

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_value(self) -> List[Any]:
        return []

    def record(self, value: Any) -> None:
        self._values.append(value)

    def summary(self) -> Optional[str]:
        if not self._values:
            return None
        return f"{self.name}: {self._values}"


class CounterMetric(Metric):
    """Metric that counts occurrences."""

    def __init__(self, name: str):
        self._name = name
        self._count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_value(self) -> int:
        return 0

    def record(self, value: Optional[Any] = None) -> None:
        self._count += 1

    def summary(self) -> Optional[str]:
        if self._count == 0:
            return None
        return f"{self.name}: {self._count}"


class ValueMetric(Metric):
    """Metric that stores a single value."""

    def __init__(self, name: str, display_format: str = "{}"):
        self._name = name
        self._value = None
        self._display_format = display_format

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_value(self) -> Any:
        return None

    def record(self, value: Any) -> None:
        self._value = value

    def summary(self) -> Optional[str]:
        if self._value is None:
            return None
        if isinstance(self._value, float):
            return f"{self.name}: {self._value:.2f}s"
        return f"{self.name}: {self._value}"


class AverageMetric(Metric):
    """Metric that computes average of recorded values."""

    def __init__(self, name: str, display_format: str = "{:.4f}"):
        self._name = name
        self._values: List[float] = []
        self._display_format = display_format

    @property
    def name(self) -> str:
        return self._name

    @property
    def default_value(self) -> List[float]:
        return []

    def record(self, value: float) -> None:
        self._values.append(value)

    def summary(self) -> Optional[str]:
        if not self._values:
            return None
        avg = sum(self._values) / len(self._values)
        return f"{self.name}: {self._display_format.format(avg)}"


class MetricsManager:
    """Manager for collecting and reporting metrics."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._metric_names: List[str] = []

    def register(self, metric: Metric) -> None:
        """Register a metric.

        Args:
            metric: Metric instance to register.
        """
        self._metrics[metric.name] = metric
        self._metric_names.append(metric.name)

    def record(self, metric_name: str, value: Optional[Any] = None) -> None:
        """Record a value for a metric.

        Args:
            metric_name: Name of the metric.
            value: Value to record.
        """
        if metric_name in self._metrics:
            self._metrics[metric_name].record(value)

    def get(self, metric_name: str) -> Any:
        """Get the current value of a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Current value or None if not found.
        """
        if metric_name in self._metrics:
            return self._metrics[metric_name].default_value
        return None

    def print_summary(self) -> None:
        """Print a summary of all collected metrics."""
        print("\n" + "=" * 50)
        print("Metrics Summary")
        print("=" * 50)
        for name in self._metric_names:
            metric = self._metrics[name]
            summary = metric.summary()
            if summary:
                print(summary)
        print("=" * 50)


# Global metrics manager instance
metrics_manager = MetricsManager()

# Register built-in metrics
metrics_manager.register(ListMetric("Merged Frame IDs"))
metrics_manager.register(AverageMetric("Average Merge Similarity"))
metrics_manager.register(CounterMetric("Total Merge Count"))
metrics_manager.register(ValueMetric("Inference Time"))
metrics_manager.register(ValueMetric("Key Frame Avg Frames Count"))
metrics_manager.register(ValueMetric("Key Frame Median Frames Count"))
metrics_manager.register(ValueMetric("Buffer Frame Count"))
metrics_manager.register(ValueMetric("Merged Embedding Count"))
metrics_manager.register(ValueMetric("Selected Detailed Frame IDs"))
metrics_manager.register(ValueMetric("Select Cost"))
