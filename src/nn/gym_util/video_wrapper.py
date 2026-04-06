import gymnasium as gym
import numpy as np
import wandb
import cv2


class VideoWrapper(gym.Wrapper):
    def __init__(
        self, 
        env,
        enabled=True, 
        steps_per_render=1,
        **kwargs
    ):
        super().__init__(env)

        self.enabled = enabled
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render

        self.frames = list()
        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        if self.enabled:
            frame = self.env.render( **self.render_kwargs)
            self.frames.append(frame)
        return obs

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.enabled and ((self.step_count % self.steps_per_render) == 0):
            frame = self.env.render(**self.render_kwargs)
            self.frames.append(frame)
        return result

    def render(self, **kwargs):
        if self.enabled:
            return np.array(self.frames)
        else:
            return None
        