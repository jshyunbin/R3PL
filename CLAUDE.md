# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep reasoning policy for robot manipulation. Applies TRM (Tiny Recursive Models) concepts to ACT (Action Chunking with Transformers) for robotic tasks using Robomimic environments (lift, can, square, stack_color) and Push-T.

Two primary models:
- **ACT**: VAE-based transformer policy (encoder compresses actions → latent z, decoder takes vision + z → action chunk)
- **ACTRM**: ACT extended with iterative deep reasoning (H-cycles × L-cycles in the decoder)

## Commands

Uses `uv` as the package manager. Python 3.12.1 required.

> **Note:** Do not run training or test commands on the user's behalf. The user runs these themselves.

**Training:**
```bash
uv run python src/nn/train.py experiment=act_stack_color
# Other experiments: act_can, actrm_can, act_lift, actrm_lift, act_square, diffusion_pusht
# Override params inline: experiment=act_can trainer.max_epochs=200 data.batch_size=64
```

**Evaluation:**
```bash
uv run python scripts/evaluate_rollout.py <config_name>
```

**Tests:**
```bash
uv run pytest
uv run pytest tests/src/nn/data/robomimic_test.py  # single test file
```

**Code quality:**
```bash
uv run task check-style   # ruff lint check
uv run task format        # ruff format
```

## Architecture

### Config System (Hydra)
All experiments defined in `src/nn/configs/`. The `experiment/` group composes `model/`, `data/`, trainer, and callbacks into a single config. Override any param on the CLI with `key=value`.

Key config groups: `model/`, `data/`, `experiment/`, `trainer/`, `logger/`, `callbacks/`.

### Model Architecture (`src/nn/models/`)

**ACT** (`act.py` → `ACTModule`):
- Manual optimization (no `automatic_optimization`)
- Encoder: joint states + action chunk → `CLS` token through `Transformer` → μ, log σ (VAE)
- Decoder: ResNet18 (pretrained, GroupNorm) vision features + latent z + joint states → `TransformerDecoder` → action predictions
- `LinearNormalizer` normalizes all inputs/outputs; fit during `setup()` from datamodule stats

**ACTRM** (`actrm.py` → `ACTRMModule`):
- Same encoder as ACT
- Decoder iterates `N_supervision` times (random 1–16 during train, fixed during val) before predicting actions
- Maintains `z_H` (high-level) and `z_L` (low-level) state vectors across reasoning cycles

### Core Modules (`src/nn/modules/trm_block.py`)
- `Transformer` / `TransformerDecoder`: standard multi-head attention blocks
- `CastedLinear`, `CastedEmbedding`, `CastedLayerNorm`: bfloat16-safe wrappers
- `RMSNorm`, `LinearSwish` (SwiGLU): used throughout

### Data Pipeline (`src/nn/data/robomimic_datamodule.py`)
- Reads Robomimic HDF5 datasets, caches as zarr for fast access
- `RobomimicReplayImageDataset`: returns sequences of images + joint states + actions
- Joint states can use rotation_6d representation (converted from axis-angle at load time)
- Normalizer stats (min/max or mean/std) computed from training split and passed to model

### Vision (`src/nn/vision/model_getter.py`)
- ResNet18 with IMAGENET1K_V1 pretrained weights; BatchNorm replaced with GroupNorm
- Lower learning rate applied to vision backbone (separate param group)

### Callbacks (`src/nn/callbacks/`)
- `RobomimicRunner`: runs full rollout episodes in Robomimic env at validation time, logs success rate to W&B
- `EMACallback`: maintains exponential moving average of model weights

## Robothink Environments

Always `import src.robothink.envs` before instantiating any Robomimic-based environment (including all scripts that use `EnvUtils`, `RobomimicImageWrapper`, or rollout runners). This import registers custom robothink environments with Robomimic; omitting it causes environment creation to fail.

## Key Conventions

- All models are `LightningModule` subclasses; training uses PyTorch Lightning + Hydra
- Logging via W&B (`wandb` logger); checkpoints saved under `train/` by default
- Data stored as HDF5 under `data/`; evaluation videos under `eval_output/`
- Rotation representation: axis-angle in dataset → converted to `rotation_6d` for model I/O

## Observation Indexing Design (Important)

Training batches and inference inputs have different observation structures — this is **intentional**:

- **Training** (`decode_action`): The batch contains a full temporal window starting from the current timestep. `obs[:, 0, ...]` correctly selects the actual current observation because index 0 is the observation at which the action was taken. The remaining timesteps are future observations within the sequence.
- **Inference** (`predict_action`): The input contains only past observations up to the current timestep (an observation buffer). `obs[:, -1, ...]` correctly selects the most recent (current) observation.

Both index the same semantic "current observation" — just from opposite ends of their respective windows. Do **not** treat this as a bug or inconsistency.
