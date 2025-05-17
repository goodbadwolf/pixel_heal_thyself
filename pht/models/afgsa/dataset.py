"""AFGSA dataset."""

import os

import h5py
import numpy as np
import torch

from pht.logger import logger


class Dataset(torch.utils.data.Dataset):
    """AFGSA dataset."""

    def __init__(self, dataset_path: str) -> None:
        assert dataset_path.endswith(".h5"), (
            "dataset_path must be the path to a .h5 file"
        )
        assert os.path.exists(dataset_path), "dataset_path is wrong"
        self.dataset_path = dataset_path
        self.dataset = None
        self.dataset_len = 0
        self.keys = []

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset."""
        with h5py.File(self.dataset_path, "r") as file:
            self.dataset_len = len(file["aux"])
            name = self.dataset_path[self.dataset_path.rfind("\\") + 1 :]
        logger.info(f"Data set {name} has {self.dataset_len} samples in total")

    def __len__(self) -> int:
        """Get dataset length."""
        return self.dataset_len

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Get item."""
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path, "r")
            self.keys = self.dataset.keys()
        data = {}
        for key in self.keys:
            data[key] = self.dataset[key][index]
        return data
