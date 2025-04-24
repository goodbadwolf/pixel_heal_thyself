import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    if cfg.model._target_.endswith("AFGSANet"):
        from pht.models.afgsa import train as afgsa_train_mod

        afgsa_train_mod.run(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model._target_}")


if __name__ == "__main__":
    main()
