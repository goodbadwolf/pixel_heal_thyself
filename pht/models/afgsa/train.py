import torch
from torch.nn import Module
from omegaconf import DictConfig

from pht.models.afgsa.model import AFGSANet, CurveOrder
from pht.models.afgsa.util import create_folder, set_global_seed
from pht.models.base_trainer import BaseTrainer, device


class AFGSATrainer(BaseTrainer):
    """Trainer class for AFGSA model."""
    
    def create_generator(self) -> Module:
        """Create and return the AFGSA generator model.
        
        Returns:
            AFGSANet model instance
        """
        padding_mode = "replicate" if self.deterministic else "reflect"
        return AFGSANet(
            self.cfg.model.in_ch,
            self.cfg.model.aux_in_ch,
            self.cfg.model.base_ch,
            num_sa=self.cfg.model.num_sa,
            block_size=self.cfg.model.block_size,
            halo_size=self.cfg.model.halo_size,
            num_heads=self.cfg.model.num_heads,
            num_gcp=self.cfg.trainer.num_gradient_checkpoint,
            padding_mode=padding_mode,
            curve_order=self.cfg.trainer.curve_order,
            use_film=self.cfg.trainer.use_film
        ).to(device)


def run(cfg: DictConfig) -> None:
    """
    Entry point for AFGSA training from Hydra.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert curve order string to enum
    cfg.trainer.curve_order = CurveOrder(cfg.trainer.curve_order)

    # Create output directories
    create_folder(cfg.paths.out_dir)
    create_folder(cfg.data.patches.root)
    
    # Create and run trainer
    trainer = AFGSATrainer(cfg)
    trainer.train()


# expose for Hydra entrypoint
__all__ = ["run"]
