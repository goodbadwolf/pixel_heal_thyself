"""PHT training script."""

import hydra
from omegaconf import DictConfig

from pht.config.registry import ConfigRegistry
from pht.hydra.plugins.pht_run_dirs_resolver import register_pht_run_dirs_resolver
from pht.logger import logger
from pht.models.afgsa.model import CurveOrder
from pht.models.afgsa.train import AFGSATrainer

register_pht_run_dirs_resolver()


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(hydra_cfg: DictConfig) -> None:
    """Train PHT model."""
    # Convert Hydra DictConfig to typed Config
    cfg = ConfigRegistry.create_config(hydra_cfg)
    logger.setup_logger(cfg.logging.level)

    # Convert curve order string to enum
    cfg.model.curve_order = CurveOrder(cfg.model.curve_order)

    # Create trainer based on model type
    if cfg.model.name == "afgsa":
        trainer = AFGSATrainer(cfg)
    elif cfg.model.name == "mamba":
        from pht.models.mamba.train import MambaTrainer
        trainer = MambaTrainer(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    trainer.train()


if __name__ == "__main__":
    main()
