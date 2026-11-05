"""Base dataset loader interface."""

from abc import ABC, abstractmethod
from pathlib import Path

class BaseDatasetLoader(ABC):
    """Abstract base class for federated medical dataset loaders."""

    def __init__(self, root_dir: str | Path, partition_csv: str | Path | None = None):
        self.root_dir = Path(root_dir)
        self.partition_csv = Path(partition_csv) if partition_csv else None

    @abstractmethod
    def load_partition(self, partition_id: int, batch_size: int = 1):
        """Return (train_loader, val_loader) for a specific client institution."""
        pass

    @abstractmethod
    def load_partition_test(self, partition_id: int, batch_size: int = 1):
        """Return DataLoader for the unseen test set of a specific client institution."""
        pass

    @abstractmethod
    def load_global_test(self, batch_size: int = 1):
        """Return DataLoader for the unseen centralized/global test set."""
        pass
