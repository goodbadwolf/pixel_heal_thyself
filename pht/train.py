import hydra
from omegaconf import DictConfig

from pht.config.registry import ConfigRegistry
from pht.hydra.plugins.pht_run_dirs_resolver import register_pht_run_dirs_resolver
from pht.models.afgsa.model import CurveOrder
from pht.models.afgsa.train import AFGSATrainer
from pht.models.mamba.train import MambaTrainer
from pht.logger import setup_logger

register_pht_run_dirs_resolver()
setup_logger()


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(hydra_cfg: DictConfig) -> None:
    # Convert Hydra DictConfig to typed Config
    cfg = ConfigRegistry.create_config(hydra_cfg)
    setup_logger(cfg.logging.level)

    # Convert curve order string to enum
    cfg.model.curve_order = CurveOrder(cfg.model.curve_order)

    # Create trainer based on model type
    if cfg.model.name == "afgsa":
        trainer = AFGSATrainer(cfg)
    elif cfg.model.name == "mamba":
        trainer = MambaTrainer(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    trainer.train()


if __name__ == "__main__":
    main()
