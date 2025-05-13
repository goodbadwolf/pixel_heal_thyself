import os
import re
from typing import Any

from omegaconf import DictConfig


class CustomRunDirs:
    def __init__(self) -> None:
        pass

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._setup_directories(config)

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        pass

    def _setup_directories(self, config: DictConfig) -> None:
        cwd = os.getcwd()
        orig_cwd = os.path.abspath(
            os.path.join(cwd, os.path.dirname(config.hydra.job.config_name))
        )

        model_name = config.model.name
        resize_value = config.data.get("resize", 1.0)
        base_pattern = f"{model_name}_p{config.data.patches.patch_size}_n{config.data.patches.num_patches}_r{resize_value}"

        is_multirun = "multirun" in cwd
        if is_multirun:
            base_dir = os.path.join(orig_cwd, "multiruns", base_pattern)
            prefix = "mrun"
        else:
            base_dir = os.path.join(orig_cwd, "runs", base_pattern)
            prefix = "run"

        os.makedirs(base_dir, exist_ok=True)

        run_num = config.get("run_num", -1)
        if run_num != -1:
            next_num = run_num
        else:
            highest_num = -1
            for item in os.listdir(base_dir):
                match = re.match(rf"{prefix}(\d+)", item)
                if match:
                    num = int(match.group(1))
                    if num > highest_num:
                        highest_num = num
            next_num = highest_num + 1
        # Create the next run/sweep directory
        output_dir = os.path.join(base_dir, f"{prefix}{next_num:03d}")
        os.makedirs(output_dir, exist_ok=True)

        if is_multirun:
            config.hydra.sweep.subdir = output_dir
        else:
            config.hydra.run.dir = output_dir
        config.paths.output_dir = output_dir
        print(f"Hydra run dir set to: {output_dir}")
