# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep reasoning policy for robot manipulation. Applies generative recursive model concepts to ACT (Action Chunking with Transformers) for robotic tasks using Robomimic environments (lift, can, square, stack_color) and Push-T.

Three primary models:
- **ACT**: VAE-based transformer policy (encoder compresses actions → latent z, decoder takes vision + z → action chunk)
- **ACTRM**: ACT extended with iterative deep reasoning (H-cycles × L-cycles in the decoder)
- **R3P** (Recursive Reasoning Robot Policy): ACT extended with GRAM (Generative Recursive Autonomous Model) applied to the transformer decoder for continuous robot actions — same architectural insertion point as ACTRM but using GRAM's generative recursive mechanism

## Commands

Uses `uv` as the package manager. Python 3.12.1 required.

**Training:**
```bash
uv run python src/nn/train.py experiment=act_stack_color
# Other experiments: act_can, actrm_can, r3p_can, act_lift, actrm_lift, r3p_lift, act_square, diffusion_pusht, act_pusht_multigoal
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

**R3P** (`r3p.py` → `R3PModule`) _(planned)_:
- Same encoder as ACT (VAE: joint states + action chunk → latent z)
- GRAM applied to the transformer decoder: generative recursive mechanism iterates over the decoder to produce continuous action outputs
- Follows the same insertion pattern as ACTRM (recursive cycles in the decoder), but replaces TRM-style H/L cycles with GRAM's generative recursion
- Targets continuous robot actions (as opposed to chunked discrete predictions)

### Core Modules (`src/nn/modules/trm_block.py`)
- `Transformer` / `TransformerDecoder`: standard multi-head attention blocks
- `CastedLinear`, `CastedEmbedding`, `CastedLayerNorm`: bfloat16-safe wrappers
- `RMSNorm`, `LinearSwish` (SwiGLU): used throughout

### Data Pipeline (`src/nn/data/robomimic_datamodule.py`)
- Reads Robomimic HDF5 datasets, caches as zarr for fast access
- `RobomimicReplayImageDataset`: returns sequences of images + joint states + actions
- Joint states can use rotation_6d representation (converted from axis-angle at load time)
- Normalizer stats (min/max or mean/std) computed from training split and passed to model

### PushT Multigoal Dataset (`src/nn/data/pusht_multitask_datamodule.py`)
- Zarr-based dataset at `data/pusht_multitask`
- Observations: `image` (96×96 RGB), `agent_pos` (2D), `keypoint` (9 T-block keypoints → flattened to 18D), `n_contacts` (1D)
- Actions: 2D continuous movement commands
- Splits: configurable `val_ratio` / `test_ratio` (default 10% each); no overlap enforced
- Config: `src/nn/configs/data/pusht_multitask.yaml` → `PushtMultitaskDataModule`
- Eval callback: `PushTMultigoalRunner` (`src/nn/callbacks/pusht_multigoal_runner.py`) — runs rollouts with `fix_goal=False` (random goal per episode), logs per-seed max rewards and videos to W&B every N epochs
- Experiment config: `src/nn/configs/experiment/act_pusht_multigoal.yaml`; monitors `val/test_success_rate`, saves top-2 checkpoints

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

## PushT Environment Notes

### PushTImageEnv (`src/nn/env/pusht_image_env.py`)
- Wraps `gym_pusht.envs.pusht.PushTEnv` directly (not via `gym.make()`) to avoid `TimeLimit`/`OrderEnforcing` wrappers
- `fix_goal=False` randomizes goal pose per episode (x/y in [100,400], angle in [-π,π]) using the env's seeded RNG after `reset()`
- `fix_goal=True` (default) keeps the fixed goal at (256, 256, π/4) — use for single-goal baselines
- **Do not use pretrained ResNet for PushT** (`pretrained_backbone: false`): ImageNet weights are inappropriate for synthetic 96×96 renders

### State Initialization Bug (angle-first order)
When replaying dataset states into `gym_pusht`, **always set angle before position**:
```python
self._env.block.angle = float(state[4])    # angle FIRST
self._env.block.position = list(state[2:4]) # then position
```
`gym_pusht._set_state()` uses legacy position-before-angle order which shifts the block due to center-of-gravity offset. The dataset was collected with angle-first, so manual initialization is required for accurate replay.

### pusht_multitask dataset
Goal pose is **not stored** in the dataset — it's only visible in rendered `img` observations. Do not try to reconstruct goal state from stored fields.

### num_workers=0 with AsyncVectorEnv
Always set `num_workers: 0` in dataloader configs when the training uses `AsyncVectorEnv`-based callbacks (e.g., `PushTMultigoalRunner`). Fork-based `AsyncVectorEnv` workers conflict with spawn-based PyTorch dataloader workers and cause silent crashes under `nohup` with no Python traceback.

### wandb
Currently configured for offline mode (`offline: true` in `src/nn/configs/logger/wandb.yaml`). To sync a completed run: `wandb sync <run_dir>/wandb/offline-run-*`

## Observation Indexing Design (Important)

Training batches and inference inputs have different observation structures — this is **intentional**:

- **Training** (`decode_action`): The batch contains a full temporal window starting from the current timestep. `obs[:, 0, ...]` correctly selects the actual current observation because index 0 is the observation at which the action was taken. The remaining timesteps are future observations within the sequence.
- **Inference** (`predict_action`): The input contains only past observations up to the current timestep (an observation buffer). `obs[:, -1, ...]` correctly selects the most recent (current) observation.

Both index the same semantic "current observation" — just from opposite ends of their respective windows. Do **not** treat this as a bug or inconsistency.
