"""Configuration registry for PHT models."""

from typing import Dict, Type

from omegaconf import DictConfig

from pht.config.base import (
    AFGSAModelConfig,
    BaseModelConfig,
    Config,
    MambaModelConfig,
)


class ConfigRegistry:
    """Registry for model configurations."""

    _model_configs: Dict[str, Type[BaseModelConfig]] = {
        "afgsa": AFGSAModelConfig,
        "mamba": MambaModelConfig,
    }

    @classmethod
    def get_model_config_class(cls, model_name: str) -> Type[BaseModelConfig]:
        """Get the configuration class for a given model name."""
        if model_name not in cls._model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        return cls._model_configs[model_name]

    @classmethod
    def register_model_config(
        cls, name: str, config_class: Type[BaseModelConfig]
    ) -> None:
        """Register a new model configuration class."""
        cls._model_configs[name] = config_class

    @classmethod
    def create_config(cls, hydra_config: DictConfig) -> Config:
        """Create a typed configuration from Hydra's DictConfig."""
        return Config.from_hydra(hydra_config)

    @classmethod
    def validate_config(cls, config: Config) -> bool:
        """Validate the configuration."""
        # Ensure model config is of the right type
        model_class = cls.get_model_config_class(config.model.name)
        if not isinstance(config.model, model_class):
            raise TypeError(
                f"Expected model config of type {model_class.__name__}, got {type(config.model).__name__}"
            )
        return True
