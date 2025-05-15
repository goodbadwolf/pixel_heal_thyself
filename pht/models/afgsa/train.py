import torch
from torch.nn import Module
from omegaconf import DictConfig

from pht.config.base import AFGSAModelConfig
from pht.models.afgsa.model import AFGSANet, CurveOrder
from pht.models.afgsa.util import create_folder
from pht.models.base_trainer import BaseTrainer, device


class AFGSATrainer(BaseTrainer):
    """Trainer class for AFGSA model."""
    
    def create_generator(self) -> Module:
        """Create and return the AFGSA generator model.
        
        Returns:
            AFGSANet model instance
        """
        model_cfg = self.cfg.model
        assert isinstance(model_cfg, AFGSAModelConfig)
        
        return AFGSANet(
            model_cfg.input_channels,
            model_cfg.aux_input_channels,
            model_cfg.feature_map_channels,
            num_sa=model_cfg.self_attention.num_layers,
            block_size=model_cfg.self_attention.block_size,
            halo_size=model_cfg.self_attention.halo_size,
            num_heads=model_cfg.self_attention.num_heads,
            num_gcp=model_cfg.num_gradient_checkpoints,
            padding_mode=self.padding_mode,
            curve_order=model_cfg.curve_order,
            use_film=model_cfg.use_film,
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
    create_folder(cfg.paths.output_dir)
    create_folder(cfg.data.patches.dir)
    
    # Create and run trainer
    trainer = AFGSATrainer(cfg)
    trainer.train()


# expose for Hydra entrypoint
__all__ = ["run"]
