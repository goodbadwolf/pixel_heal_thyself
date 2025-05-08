import hydra
from omegaconf import DictConfig
import torch

from pht.models.afgsa.train import AFGSATrainer
from pht.models.mamba.train import MambaTrainer


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    if cfg.model.name == "afgsa":
        trainer = AFGSATrainer(cfg)
    elif cfg.model.name == "mamba":
        trainer = MambaTrainer(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    trainer.train()


if __name__ == "__main__":
    main()
