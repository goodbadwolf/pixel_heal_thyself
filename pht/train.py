import hydra
from omegaconf import DictConfig
import torch


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    # dispatch to model‑specific training
    if cfg.model._target_.endswith("AFGSANet"):
        from pht.models.afgsa import train as afgsa_train_mod

        afgsa_train_mod.run(cfg)
    else:
        raise ValueError(f"Unsupported model: {cfg.model._target_}")


if __name__ == "__main__":
    main()
