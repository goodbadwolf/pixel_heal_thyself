"""Hydra plugin for PHT run directories."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from pht.logger import logger
from pht.utils import SingletonMeta, is_none_or_empty, is_truthy


@dataclass
class PhtRunDirsCache(metaclass=SingletonMeta):
    """
    Singleton cache for storing the last used run_dir and is_multirun state.

    Used between multiple calls to the `pht_run_dirs_resolver` in a single process.
    """

    is_multirun: Optional[bool] = None
    run_dir: Optional[Path] = None

    def reset(self) -> None:
        """Reset the cache to its initial state."""
        self.is_multirun = None
        self.run_dir = None


def pht_run_dirs_resolver(
    cfg_output_root_dir: Optional[str] = "outputs",
    cfg_base_pattern: Optional[str] = None,
    cfg_job_subdir: Optional[str] = None,
    cfg_run_num: Optional[str] = "-1",
    cfg_is_multirun: Optional[str] = "False",
) -> str:
    """
    OmegaConf custom resolver for generating and managing output directories for runs and sweeps/multirun.

    Args:
        cfg_output_root_dir: Root directory for all outputs (default: 'outputs').
        cfg_base_pattern: Base pattern for naming the run directory.
        cfg_job_subdir: Optional subdirectory for jobs (used in sweeps/multirun).
        cfg_run_num: Run number (if -1, auto-increment).
        cfg_is_multirun: Whether this is a instance of a trial/multirun (truthy/falsy).

    Returns:
        str: Path of the created run directory, relative to the current working directory.

    """
    cache = PhtRunDirsCache()

    cwd = Path.cwd()
    is_multirun = is_truthy(cfg_is_multirun)
    job_subdir = None if is_none_or_empty(cfg_job_subdir) else Path(cfg_job_subdir)
    base_pattern = (
        None if is_none_or_empty(cfg_base_pattern) else Path(cfg_base_pattern)
    )

    try:
        run_num = int(cfg_run_num)
    except Exception:
        run_num = -1

    if is_multirun and cache.is_multirun is None:
        cache.is_multirun = True

    if is_multirun:
        base_dir = cwd / cfg_output_root_dir / "trials"
    else:
        base_dir = cwd / cfg_output_root_dir / "runs" / base_pattern

    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_num = int(run_num)
    except Exception:
        run_num = -1

    if run_num != -1:
        next_num = run_num
    else:
        highest_num = -1
        for item in base_dir.iterdir():
            pattern = r"run(\d+)"
            match = re.match(pattern, item.name)
            if match:
                num = int(match.group(1))
                highest_num = max(highest_num, num)
        next_num = highest_num + 1

    run_dir: Path = base_dir / f"run{next_num:03d}"

    if not is_none_or_empty(cache.run_dir):
        run_dir = cache.run_dir
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cache.run_dir = run_dir

    if not is_none_or_empty(job_subdir) and cache.is_multirun:
        run_dir = run_dir / job_subdir

    run_dir.mkdir(parents=True, exist_ok=True)

    run_dir_str = str(run_dir.relative_to(cwd))
    logger.info(f"Created run directory: {run_dir_str}")
    return run_dir_str


def register_pht_run_dirs_resolver() -> None:
    """Register the pht_run_dirs_resolver as an OmegaConf custom resolver."""
    OmegaConf.register_new_resolver("pht_run_dirs", pht_run_dirs_resolver, replace=True)


def reset_pht_run_dirs_cache() -> None:
    """Reset the singleton cache for the run dirs resolver."""
    PhtRunDirsCache().reset()
