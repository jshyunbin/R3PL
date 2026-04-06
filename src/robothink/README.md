# Custom Robosuite Environments

Custom robosuite environments for data collection


## Introduction

You can easily collect datasets with custom environments by using the below command
```zsh
uv run mjpython scripts/collect_demonstrations.py --directory data/ --environment StackThreeCubes --device spacemouse
```
> ⚠️ `mjpython` is specifically for mac. For Linux and others replace `mjpython` with `python`

You can also check the collected datasets by using the following command
```zsh
uv run python scripts/playback_demonstrations.py --folder data/{file_name}
```

## Requirements

Install required python packages by running the below command. `uv` is recommended but not necessary. 

```zsh
uv venv -p $(which python3.10)
uv pip install requirements.txt
```

Tested on MacBook Pro M5 Tahoe 26.3

## List of Environments

### StackThreeCubes
An environment similar to the robosuite's `Lift` preset but with two more cubes in different colors. 

This environment's goal is to move each cube so that the green cube is stacked above the red cube and the blue cube is stacked above the green cube. This environment is challenging due to the need to understand the current state and decide which cube to pick up. 

The reward will only be triggered when all three cubes are stacked in the right orientation. Therefore it is not only important to stack all three cubes stable but also stack them in the right order. 

## Helpful Scripts

Convert robosuite dataset to robomimic dataset
```zsh
uv run python -m robomimic.scripts.conversion.convert_robosuite --dataset /path/to/dataset.hdf5
```

Extract observations from MuJoCo states
```zsh
uv run python scripts/dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name output_name.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 256 --camera_width 256 --done_mode 2
```