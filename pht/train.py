import hydra
from omegaconf import DictConfig
import torch
from hydra.utils import instantiate


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    # instantiate model
    model = instantiate(cfg.model)

    # instantiate optimizer & scheduler
    optimizer = instantiate(cfg.trainer.optim, params=model.parameters())
    scheduler = instantiate(cfg.trainer.scheduler, optimizer=optimizer)

    # dispatch to model‑specific training
    if cfg.model._target_.endswith("AFGSANet"):
        from pht.models.afgsa import train as afgsa_train_mod

        afgsa_train_mod.run(cfg, model, optimizer, scheduler)
    else:
        raise ValueError(f"Unsupported model: {cfg.model._target_}")


if __name__ == "__main__":
    main()
