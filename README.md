# Pixel Heal Thyself (PHT)

Deep-learning based denoiser for volumetric path traced images.

## Installation

```bash
git clone https://github.com/goodbadwolf/pixel_heal_thyself.git
cd pixel_heal_thyself
uv install
```

## Training

PHT uses Hydra for configuration management. From the root of the repository, run:

```bash
uv run python -m pht.train [overrides]
```

### Examples

- Full dataset, default settings:

```bash
uv run python -m pht.train
```

- Small dataset preset, 8 epochs:

```bash
uv run python -m pht.train dataset=small trainer.epochs=8
```

## Configuration

- `config/config.yaml`: Global defaults
- `config/model/`: Model parameters (e.g. `afgsa.yaml`)
- `config/data/`: Dataset presets (`default`, `small`, `limited`)
- `config/trainer/`: Training parameters (batch size, epochs, etc.)
