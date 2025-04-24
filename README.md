# Pixel Heal Thyself (PHT)

Deep-learning based denoiser for volumetric path traced images.

## Installation

```bash
git clone https://github.com/goodbadwolf/pixel_heal_thyself.git
cd pixel_heal_thyself
uv install
```

## Training

### Available Configurations

PHT uses Hydra for configuration management. From the root of the repository, run:

```bash
uv run python -m pht.train -cn <config name> [overrides]
```

where `<config name>` is the name of the configuration file in `config/` (without the `.yaml` extension).

The available configuration names are:

- `ci`: To be used for continuous integration testing
- `dev`: Trains on patches of size 32x32, 100 patches per image. To be used for development and debugging.
- `stag`: Trains on patches of size 64x64, 200 patches per image. To be used for staging and testing.
- `prod`: Trains on patches of size 128x128, 400 patches per image. To be used for actual training and experimentation.

### Overrides

You can override any configuration parameter by appending it to the command.

For example, to set the number of epochs to 8, you can run:

```bash
uv run python -m pht.train -cn dev trainer.epochs=8
```

## Datasets

**TBD**
