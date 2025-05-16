"""Base configuration classes for PHT."""

from dataclasses import dataclass, field
from typing import List, Union

from omegaconf import DictConfig

from pht.models.afgsa.model import CurveOrder


@dataclass
class PathConfig:
    """Configuration for paths."""

    root: str = "${hydra:runtime.cwd}"
    output_dir: str = "${hydra:run.dir}"


@dataclass
class ImagesConfig:
    """Configuration for images."""

    dir: str = "${paths.root}/data/images"
    scale: float = 1.0


@dataclass
class PatchesConfig:
    """Configuration for data patches."""

    patch_size: int = 128
    num_patches: int = 400
    dir: str = "${data.images.dir}/patches_${data.patches.patch_size}_n${data.patches.num_patches}_r${data.images.scale}"


@dataclass
class DataConfig:
    """Configuration for data."""

    images: ImagesConfig = field(default_factory=ImagesConfig)
    patches: PatchesConfig = field(default_factory=PatchesConfig)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    _target_: str = "torch.optim.Adam"
    lr: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Configuration for scheduler."""

    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    milestones: List[int] = field(default_factory=lambda: [3, 6, 9])
    gamma: float = 0.5


@dataclass
class LossesConfig:
    """Configuration for model losses."""

    l1_loss_w: float = 1.0
    gan_loss_w: float = 0.005
    gp_loss_w: float = 10
    use_lpips_loss: bool = False
    lpips_loss_w: float = 0.1
    use_ssim_loss: bool = False
    ssim_loss_w: float = 0.1


@dataclass
class TrainerConfig:
    """Configuration for trainer."""

    # Training settings
    batch_size: int = 8
    epochs: int = 12
    deterministic: bool = True
    save_interval: int = 1
    num_saved_imgs: int = 6

    # Optimizer and scheduler
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Learning rates
    lrG: float = 1e-4
    lrD: float = 1e-4
    lr_gamma: float = 0.5
    lr_milestone: int = 3

    load_model: bool = False


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention blocks."""

    num_layers: int = 5
    block_size: int = 8
    halo_size: int = 3
    num_heads: int = 4


@dataclass
class DiscriminatorConfig:
    """Configuration for discriminator."""

    use_multiscale_discriminator: bool = False
    use_film: bool = False


@dataclass
class BaseModelConfig:
    """Base configuration for all models."""

    name: str
    input_channels: int = 3
    aux_input_channels: int = 7
    feature_map_channels: int = 256
    curve_order: CurveOrder = CurveOrder.RASTER
    use_film: bool = False
    num_gradient_checkpoints: int = 0
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    losses: LossesConfig = field(default_factory=LossesConfig)


@dataclass
class SelfAttentionConfig:
    """Configuration for self-attention blocks."""

    num_layers: int = 5
    block_size: int = 8
    halo_size: int = 3
    num_heads: int = 4


@dataclass
class AFGSAModelConfig(BaseModelConfig):
    """Configuration for AFGSA model."""

    name: str = "afgsa"
    self_attention: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)


@dataclass
class MambaModelConfig(BaseModelConfig):
    """Configuration for Mamba model."""

    name: str = "mamba"
    num_layers: int = 5
    d_state: int = 64
    d_conv: int = 4
    expansion: int = 4


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"


@dataclass
class Config:
    """Main configuration for PHT."""

    seed: int = 990819
    data_ratio: float = 0.95
    run_num: int = -1
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Union[AFGSAModelConfig, MambaModelConfig] = field(
        default_factory=lambda: AFGSAModelConfig()
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def parse_nested_config(cls, cfg_section, config_class):
        """Utility to parse a nested config section into a dataclass instance."""
        return config_class(**{k: v for k, v in cfg_section.items()})

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "Config":
        """Create a Config instance from a Hydra DictConfig."""
        # Convert DictConfig to plain dict
        cfg_dict = {
            k: v for k, v in cfg.items() if not k.startswith("_") and k != "model"
        }

        # Create appropriate model config based on model name
        if cfg.model.name == "afgsa":
            model_cfg = AFGSAModelConfig(**{k: v for k, v in cfg.model.afgsa.items()})
        elif cfg.model.name == "mamba":
            model_cfg = MambaModelConfig(**{k: v for k, v in cfg.model.mamba.items()})
        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")

        cfg_dict["model"] = model_cfg

        # Create nested configs using the utility
        cfg_dict["paths"] = cls.parse_nested_config(cfg.paths, PathConfig)
        cfg_dict["data"] = DataConfig(
            images=cls.parse_nested_config(cfg.data.images, ImagesConfig),
            patches=cls.parse_nested_config(cfg.data.patches, PatchesConfig),
        )

        # Handle trainer config with losses subconfig
        trainer_dict = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["optim", "scheduler", "losses"]
        }
        trainer = cls.parse_nested_config(trainer_dict, TrainerConfig)
        trainer.optim = cls.parse_nested_config(cfg.trainer.optim, OptimizerConfig)
        trainer.scheduler = cls.parse_nested_config(
            cfg.trainer.scheduler, SchedulerConfig
        )

        cfg_dict["trainer"] = trainer

        cfg_dict["logging"] = cls.parse_nested_config(cfg.logging, LoggingConfig)

        # Remove keys that start with "_"
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith("_")}

        return cls(**cfg_dict)
