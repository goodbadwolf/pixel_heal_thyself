# Model Configuration Structure

This directory contains the configuration files for different model architectures and their variants.

## Directory Structure

```
config/model/
├── common.yaml           # Common parameters shared across all models
├── architectures/        # Model architecture specific configs
│   ├── afgsa.yaml       # AFGSA model configuration
│   ├── mamba.yaml       # Mamba model configuration
│   └── ...              # Future model architectures
└── variants/            # Model variants (e.g., small, large versions)
    ├── afgsa/           # AFGSA variants
    └── mamba/           # Mamba variants
```

## Adding a New Model

To add a new model architecture:

1. Create a new YAML file in the `architectures/` directory
2. Inherit common parameters using `${common.parameter_name}`
3. Define architecture-specific parameters
4. Add appropriate documentation comments

Example:

```yaml
name: new_model
version: ${common.version}

# Architecture specific parameters
num_blocks: 5 # Number of processing blocks
feature_size: 64 # Size of feature maps

# Inherited parameters
in_ch: ${common.in_ch}
feature_map_channels: ${common.feature_map_channels}
batch_size: ${common.batch_size}
```

## Adding Model Variants

To add variants of an existing model:

1. Create a new directory in `variants/<model_name>/`
2. Create variant-specific YAML files (e.g., `small.yaml`, `large.yaml`)
3. Inherit from the base architecture and override specific parameters

## Common Parameters

The `common.yaml` file contains parameters shared across all models:

### Input/Output Configuration

- `in_ch`: Number of input channels
- `aux_in_ch`: Number of auxiliary input channels
- `feature_map_channels`: Base number of channels in feature maps

### Training Parameters

- `batch_size`: Number of samples per training batch
- `learning_rate`: Initial learning rate for the optimizer
- `weight_decay`: L2 regularization weight decay

### Version Information

- `version`: Configuration schema version
