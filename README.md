# Deep Reasoning Policy model 

This is an ongoing independent research to combine deep reasoning (inspired by TRM) to action generation policy models. 

## Introduction

There are currently two models. `ACT` and `ACTRM`. `ACTRM` is a policy generating model with deep reasoning capabilities. 

You can simply train models.
```zsh
uv run python src/nn/train.py experiment=act_stack_color
```

## Models

## Dataset
Currently, we only support robosuite and robomimic datasets.

If you want to add customized robosuite dataset, consider adding it in [robothink](https://github.com/jshyunbin/robothink). 

This robothink repository is a collection of custom environments for this project.

## Useful Scripts
### Evaluate rollout

To evaluate a model trained with robomimic environment, run the below
```zsh
uv run python scripts/evaluate_rollout.py act_stack_color
```

### Visualize Trajectory

### Visualize Attention Map



## Acknowledgement

This repository is forked from [here](https://github.com/olivkoch/nano-trm).
