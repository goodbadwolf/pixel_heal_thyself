"""AFGSA HDF5 constructor."""

import multiprocessing
import os
import random
from multiprocessing import Process, Queue, Value

import h5py
import numpy as np

from pht.logger import logger
from pht.models.afgsa.preprocessing import get_cropped_patches


class Hdf5Constructor:
    """AFGSA HDF5 constructor."""

    def __init__(  # noqa: PLR0913
        self,
        data_path: str,
        save_path: str,
        patch_size: int,
        num_patches: int,
        seed: int,
        train_val_ratio: float,
        scale: float = 1.0,
        noisy_spp: int = 32,
        gt_spp: int = 1024,
        deterministic: bool = False,
    ) -> None:
        """Initialize HDF5 constructor."""
        self.data_path = data_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.seed = seed
        self.train_val_ratio = (train_val_ratio, 1 - train_val_ratio)
        self.noisy_spp = noisy_spp
        self.gt_spp = gt_spp
        self.deterministic = deterministic
        self.scale = scale

        self.exr_paths = []
        self.gt_paths = []
        self.paths = []

    def construct_hdf5(self) -> None:
        """Construct HDF5 dataset."""
        logger.info("Constructing data set (hdf5)")
        self.get_exr_paths()
        self.get_cropped_patches()
        logger.info("Constructing data set (hdf5): done")

    def get_exr_paths(self) -> None:
        """Get EXR paths."""
        logger.info("Get exr paths")
        self.noisy_path = os.path.join(self.data_path, f"{self.noisy_spp}spp")
        self.gt_path = os.path.join(self.data_path, f"{self.gt_spp}spp")
        for _root, dirs, files in os.walk(self.gt_path):
            if not dirs:
                for file in files:
                    gt_path = os.path.join(
                        self.gt_path,
                        f"{file.split('_')[0]}_{file.split('_')[1]}_{self.gt_spp}",
                    )
                    exr_path = os.path.join(
                        self.noisy_path,
                        f"{file.split('_')[0]}_{file.split('_')[1]}_{self.noisy_spp}",
                    )
                    if gt_path not in self.gt_paths:
                        self.gt_paths.append(gt_path)
                        self.exr_paths.append(exr_path)
        assert len(self.exr_paths) == len(self.gt_paths), (
            "#samples does not equal to #gts, check the data!"
        )

        if not self.deterministic:
            assert type(self.seed) is int, "seed must be an int!"
            random.seed(self.seed)

        self.paths = [
            (e, g) for e, g in zip(self.exr_paths, self.gt_paths, strict=False)
        ]
        random.shuffle(self.paths)
        logger.info("Get exr paths: done")

    def worker(  # noqa: PLR0913
        self,
        worker_id: int,
        queues: list[Queue],
        path_mapping: dict[str, str],
        name_shape_mapping: dict[str, tuple[int, int, int, int]],
        lock: multiprocessing.Lock,  # type: ignore
        v: Value,  # type: ignore
    ) -> None:
        """Worker."""
        rng = (
            random.Random(self.seed + worker_id)
            if self.deterministic
            else random.Random()
        )
        while not queues[0].empty() or not queues[1].empty():
            v.value += 1
            logger.info(f"Generating patches: {v.value} / {len(self.paths) - 3}")
            if not queues[0].empty():
                path = queues[0].get()
                dataset = "train"
            elif not queues[1].empty():
                path = queues[1].get()
                dataset = "val"
            cropped, patches = get_cropped_patches(
                path[0],
                path[1],
                self.patch_size,
                self.num_patches,
                rng,
                scale=self.scale,
            )

            lock.acquire()
            with h5py.File(path_mapping[dataset], "a") as hf:
                for key in name_shape_mapping:
                    temp = [c[key] for c in cropped]
                    hf[key].resize((hf[key].shape[0] + len(temp)), axis=0)
                    hf[key][-len(temp) :] = temp
            lock.release()

    def get_cropped_patches(self) -> None:
        """Get cropped patches."""
        rng = random.Random(self.seed) if self.deterministic else random.Random()
        patch_count = 0
        train_save_path = os.path.join(self.save_path, "train.h5")
        val_save_path = os.path.join(self.save_path, "val.h5")
        path_mapping = {"train": train_save_path, "val": val_save_path}
        name_shape_mapping = {
            "noisy": (None, self.patch_size, self.patch_size, 3),
            "gt": (None, self.patch_size, self.patch_size, 3),
            "aux": (None, self.patch_size, self.patch_size, 7),
        }
        queues = [Queue() for _ in range(2)]  # [train_queue, val_queue]
        # the first 2 paths are used to initiate h5py files
        for i in range(2, len(self.paths)):
            if (i - 2) < int(self.train_val_ratio[0] * (len(self.paths) - 2)):
                queues[0].put(self.paths[i])
            else:
                queues[1].put(self.paths[i])
        # initiate h5py files
        logger.info("Initiating h5py files")
        for i, n in enumerate(["train", "val"]):
            with h5py.File(path_mapping[n], "w") as hf:
                cropped, patches = get_cropped_patches(
                    self.paths[i][0],
                    self.paths[i][1],
                    self.patch_size,
                    self.num_patches,
                    rng,
                    scale=self.scale,
                )
                patch_count += len(cropped)
                for key in name_shape_mapping:  # noqa: PLC0206
                    temp = np.array([c[key] for c in cropped])
                    hf.create_dataset(
                        key,
                        data=temp,
                        maxshape=name_shape_mapping[key],
                        compression="gzip",
                        chunks=True,
                    )
        logger.info("Initiating h5py files: done")

        # start processes
        lock = (
            multiprocessing.Lock()
        )  # to ensure only one process writes to the file at once
        done_exr_count = Value("i", 0)
        pool = [
            Process(
                target=self.worker,
                args=(
                    i,
                    queues,
                    path_mapping,
                    name_shape_mapping,
                    lock,
                    done_exr_count,
                ),
            )
            for i in range(multiprocessing.cpu_count() - 1)
        ]
        for p in pool:
            p.start()
        for p in pool:
            p.join()

        logger.info("Generating patches: done")
