# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training

The project uses Hydra for configuration management. Training is run using Python:

```bash
# Basic training command
uv run python -m pht.train

# With configuration selection
uv run python -m pht.train -cn <config_name>

# With parameter overrides
uv run python -m pht.train -cn dev trainer.epochs=10 model=mamba
```

#### Configuration Options

- **Configs**: `dev`, `stag`, `prod`, `default`
  - `dev`: patch_size=32, num_patches=100, scale=0.5
  - `stag`: patch_size=64, num_patches=200, scale=0.5
  - `prod`: patch_size=128, num_patches=400, scale=1.0
  - `default`: patch_size=128, num_patches=400, scale=1.0

#### Model Options (use Hydra override syntax)

- **LPIPS loss**: `model.losses.use_lpips_loss=true model.losses.lpips_loss_w=VALUE`
- **SSIM loss**: `model.losses.use_ssim_loss=true model.losses.ssim_loss_w=VALUE`
- **Multiscale discriminator**: `model.discriminator.use_multiscale_discriminator=true`
- **FiLM conditioning**: `model.use_film=true`
- **Model selection**: `model=afgsa` or `model=mamba`

#### Curve Ordering

- **Raster curve** (default): `model.curve_order=raster`
- **Hilbert curve**: `model.curve_order=hilbert`
- **Z-order curve**: `model.curve_order=zorder`

#### Other Parameters

- **Epochs**: `trainer.epochs=N` (default: 12)
- **Batch size**: `trainer.batch_size=N` (default: 8)
- **Learning rates**: `model.lr_g=VALUE model.lr_d=VALUE` (default: 1e-4)

### Testing

- Evaluation happens automatically after training completes
- Results are saved in the `outputs/` directory under run-specific folders
- Each run folder contains `evaluation.txt` with metrics

## Code Style Guidelines

### Imports

- Sort imports: standard library first, third-party next, local modules last
- Use explicit imports instead of wildcard imports

### Formatting

- Using Ruff for linting (`tool.ruff` in pyproject.toml)
- 4-space indentation
- Line length: 120 characters

### Types

- Strong typing with Python type hints recommended for new code
- Function parameters should be typed when possible

### Error Handling

- Use try/except with specific exception types
- Prefer context managers (`with` statements) for resource handling

### Model Code

- Models are in PyTorch framework
- Use CUDA tensors for computational efficiency
- Neural network implementations are in discrete modules
- Available models: AFGSA and Mamba
- Models are located in `pht/models/`

### Comments

- Only use comments to explain complex logic
- Use docstrings for module-level and class-level documentation.
  Think deeply about the docstrings and make sure they capture the
  essence of the code and not just the implementation.
- Use inline comments sparingly and only when necessary
- Use TODO comments for future improvements or fixes
- Use FIXMEs for known issues that need to be addressed
- Use NOTE comments for important information or clarifications

### Variable Naming

- Use descriptive variable names, unless it makes the code unwiedly or hard to write

### Git Workflow

- Before running any git commands, ALWAYS check the current git state like branch and status. NEVER assume the state.
- When performing git actions on remote branches, like pushing or pulling, always use the same name for both local and remote branches. Never push a local branch to a remote branch with a different name unless explicitly requested