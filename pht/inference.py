import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Top-level inference dispatcher (mirrors pht/train.py).
    Choose the correct model-specific runner from cfg.model.name.
    """
    if cfg.model.name == "afgsa":
        from pht.models.afgsa import inference as afgsa_inf_mod

        afgsa_inf_mod.run(cfg)
    else:
        raise ValueError(f"Unsupported model for inference: {cfg.model.name}")


if __name__ == "__main__":
    main()
