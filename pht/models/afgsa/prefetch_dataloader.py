"""AFGSA prefetch dataloader."""

from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    """DataLoaderX."""

    def __iter__(self) -> BackgroundGenerator:
        """Iterate."""
        return BackgroundGenerator(super().__iter__())
