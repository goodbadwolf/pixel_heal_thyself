import torch
from torch.nn import Module
from omegaconf import DictConfig

from pht.models.afgsa.model import CurveOrder
from pht.models.afgsa.util import create_folder, set_global_seed
from pht.models.mamba.model import MambaDenoiserNet, PositionalEncoding2D
from pht.models.base_trainer import BaseTrainer, device


class MambaTrainer(BaseTrainer):
    """Trainer class for Mamba model."""
    
    def create_generator(self) -> Module:
        """Create and return the Mamba generator model.
        
        Returns:
            MambaDenoiserNet model instance
        """
        pos_encoder = PositionalEncoding2D(
            self.cfg.model.base_ch,
            self.cfg.data.patches.patch_size,
            self.cfg.data.patches.patch_size,
        ).to(device)

        return MambaDenoiserNet(
            self.cfg.model.in_ch,
            self.cfg.model.aux_in_ch,
            self.cfg.model.base_ch,
            pos_encoder,
            num_blocks=self.cfg.model.num_blocks,
            d_state=self.cfg.model.d_state,
            d_conv=self.cfg.model.d_conv,
            expansion=self.cfg.model.expansion,
            num_gcp=self.cfg.model.num_gcp,
        ).to(device)


def run(cfg: DictConfig) -> None:
    """
    Entry point for Mamba training from Hydra.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert curve order string to enum
    cfg.trainer.curve_order = CurveOrder(cfg.trainer.curve_order)

    # Set up deterministic training if requested
    if cfg.trainer.get("deterministic", False):
        set_global_seed(cfg.seed)

    # Create output directories
    create_folder(cfg.paths.out_dir)
    create_folder(cfg.data.patches.root)
    
    # Create and run trainer
    trainer = MambaTrainer(cfg)
    trainer.train()


# expose for Hydra entrypoint
__all__ = ["run"]
