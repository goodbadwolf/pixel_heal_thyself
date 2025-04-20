# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build and Run
- Train: `./tools/train.sh [-s|--small|-f|--full|-l|--limited] [--epochs=N]`
- Options:
  - Dataset options: `-s|--small`, `-f|--full`, `-l|--limited`
  - Model options: `--lpips-loss=VALUE`, `--ssim-loss=VALUE`, `--multiscale-discriminator`, `--use-film`
  - Curve ordering: `--raster-curve`, `--hilbert-curve`, `--zorder-curve`

### Test
- Run evaluation using the model output in the runs directory

## Code Style Guidelines

### Imports
- Sort imports: standard library first, third-party next, local modules last
- Use explicit imports instead of wildcard imports

### Formatting
- Using Ruff for linting (`tool.ruff` in pyproject.toml)
- 4-space indentation
- Line length: ~100 characters

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