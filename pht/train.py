import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    if cfg.model.name == "afgsa":
        from pht.models.afgsa import train as afgsa_train_mod

        afgsa_train_mod.run(cfg)
    elif cfg.model.name == "mamba":
        from pht.models.mamba import train as mamba_train_mod

        mamba_train_mod.run(cfg)

    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")


if __name__ == "__main__":
    main()
