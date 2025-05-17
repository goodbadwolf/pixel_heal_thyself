"""Mamba trainer."""

from omegaconf import DictConfig
from torch.nn import Module

from pht.config.base import MambaModelConfig
from pht.models.afgsa.model import CurveOrder
from pht.models.afgsa.util import create_folder
from pht.models.base_trainer import BaseTrainer, device
from pht.models.mamba.model import MambaDenoiserNet, PositionalEncoding2D


class MambaTrainer(BaseTrainer):
    """Trainer class for Mamba model."""

    def create_generator(self) -> Module:
        """
        Create and return the Mamba generator model.

        Returns:
            MambaDenoiserNet model instance

        """
        # We know the model is MambaModelConfig
        model_cfg = self.cfg.model
        assert isinstance(model_cfg, MambaModelConfig)

        pos_encoder = PositionalEncoding2D(
            model_cfg.feature_map_channels,
            self.cfg.data.patches.patch_size,
            self.cfg.data.patches.patch_size,
        ).to(device)

        return MambaDenoiserNet(
            model_cfg.input_channels,
            model_cfg.aux_input_channels,
            model_cfg.feature_map_channels,
            pos_encoder,
            num_blocks=model_cfg.num_layers,
            d_state=model_cfg.d_state,
            d_conv=model_cfg.d_conv,
            expansion=model_cfg.expansion,
            num_gcp=model_cfg.num_gradient_checkpoints,
            padding_mode=self.padding_mode,
        ).to(device)


def run(cfg: DictConfig) -> None:
    """
    Entry point for Mamba training from Hydra.

    Args:
        cfg: Hydra configuration object

    """
    # Convert curve order string to enum
    cfg.trainer.curve_order = CurveOrder(cfg.trainer.curve_order)

    # Create output directories
    create_folder(cfg.paths.output_dir)
    create_folder(cfg.data.patches.dir)

    # Create and run trainer
    trainer = MambaTrainer(cfg)
    trainer.train()


# expose for Hydra entrypoint
__all__ = ["run"]
