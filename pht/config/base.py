"""Base configuration classes for PHT."""

from dataclasses import dataclass, field
from typing import List, Union

from omegaconf import DictConfig


@dataclass
class PathConfig:
    """Configuration for paths."""

    root: str = "${hydra:runtime.cwd}"
    output_dir: str = "${hydra:run.dir}"


@dataclass
class PatchesConfig:
    """Configuration for data patches."""

    patch_size: int = 128
    num_patches: int = 400
    dir: str = "${data.images.dir}/patches_${data.patches.patch_size}_n${data.patches.num_patches}_r${data.resize}"


@dataclass
class ImagesConfig:
    """Configuration for images."""

    dir: str = "${paths.root}/data/images"
    scale: float = 1.0


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
    deterministic: bool = False
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
    curve_order: str = "raster"
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    losses: LossesConfig = field(default_factory=LossesConfig)


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
    num_gcp: int = 0


@dataclass
class ModelCommonConfig:
    """Common configuration for all models."""

    input_channels: int = 3
    aux_input_channels: int = 7
    feature_map_channels: int = 256
    curve_order: str = "raster"
    use_film: bool = False


@dataclass
class Config:
    """Main configuration for PHT."""

    seed: int = 990819
    data_ratio: float = 0.95
    run_num: int = -1
    model_common: ModelCommonConfig = field(default_factory=ModelCommonConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Union[AFGSAModelConfig, MambaModelConfig] = field(
        default_factory=lambda: AFGSAModelConfig()
    )

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "Config":
        """Create a Config instance from a Hydra DictConfig."""
        # Convert DictConfig to plain dict
        cfg_dict = {
            k: v
            for k, v in cfg.items()
            if not k.startswith("_") and k != "model" and k != "model_common"
        }

        # Create appropriate model config based on model name
        if cfg.model.name == "afgsa":
            model_cfg = AFGSAModelConfig(**{k: v for k, v in cfg.model.afgsa.items()})
        elif cfg.model.name == "mamba":
            model_cfg = MambaModelConfig(**{k: v for k, v in cfg.model.mamba.items()})
        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")

        cfg_dict["model"] = model_cfg

        # Create model_common config
        if hasattr(cfg, "model_common"):
            cfg_dict["model_common"] = ModelCommonConfig(**cfg.model_common)

        # Create nested configs
        cfg_dict["paths"] = PathConfig(**cfg.paths)
        cfg_dict["data"] = DataConfig(
            images=ImagesConfig(**cfg.data.images),
            patches=PatchesConfig(**cfg.data.patches),
        )

        # Handle trainer config with losses subconfig
        trainer_dict = {
            k: v
            for k, v in cfg.trainer.items()
            if k not in ["optim", "scheduler", "losses"]
        }
        trainer = TrainerConfig(**trainer_dict)
        trainer.optim = OptimizerConfig(**cfg.trainer.optim)
        trainer.scheduler = SchedulerConfig(**cfg.trainer.scheduler)

        cfg_dict["trainer"] = trainer

        return cls(**cfg_dict)
