import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pusht.envs.pusht import PushTEnv as _PushTEnv


class PushTImageEnv(gym.Env):
    """Image-based PushT environment wrapping gym_pusht's PushTEnv.

    Observations are returned as channel-first float32 images normalized to [0, 1],
    plus agent position. Supports a `fix_goal` parameter: when False, the goal pose
    is randomized each episode (multigoal setting).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, fix_goal: bool = True, render_size: int = 96):
        super().__init__()
        # Instantiate the underlying env directly to avoid TimeLimit/OrderEnforcing wrappers
        self._env = _PushTEnv(
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=render_size,
            observation_height=render_size,
        )
        self.fix_goal = fix_goal
        self.render_size = render_size
        self._last_frame = None

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, render_size, render_size),
                    dtype=np.float32,
                ),
                "agent_pos": spaces.Box(
                    low=0.0,
                    high=512.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        reset_to_state = options.get("reset_to_state") if options else None

        if reset_to_state is not None:
            # Call reset without the state option so _setup() runs, then set
            # state manually using angle-first order. gym_pusht's legacy _set_state
            # sets position before angle, which shifts the block due to CoG offset;
            # the dataset was collected with angle-first ordering (UVA convention).
            obs, info = self._env.reset(seed=seed)
            state = np.array(reset_to_state)
            self._env.agent.position = list(state[:2])
            self._env.block.angle = float(state[4])
            self._env.block.position = list(state[2:4])
            self._env.space.step(self._env.dt)
            obs = self._env.get_obs()
            info = self._env._get_info()
            info["is_success"] = False
        else:
            obs, info = self._env.reset(seed=seed)

        if not self.fix_goal:
            rng = self._env.np_random
            self._env.goal_pose = np.array(
                [
                    float(rng.integers(100, 400)),
                    float(rng.integers(100, 400)),
                    float(rng.uniform(-np.pi, np.pi)),
                ]
            )
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def render(self):
        # Returns the cached 96x96 uint8 (H, W, 3) observation frame.
        # Avoids the expensive high-res render; VideoRecordingWrapper only needs uint8 HWC.
        return self._last_frame

    def seed(self, seed):
        # MultiStepWrapper calls env.seed(seed) to initialize the environment.
        self._env.reset(seed=seed)

    def close(self):
        self._env.close()

    def _process_obs(self, obs):
        pixels = obs["pixels"]                            # uint8 (H, W, 3) at render_size
        self._last_frame = pixels                         # cache for render()
        image = pixels.astype(np.float32) / 255.0        # (H, W, 3) → float [0, 1]
        image = np.moveaxis(image, -1, 0)                 # HWC → CHW
        agent_pos = obs["agent_pos"].astype(np.float32)
        return {"image": image, "agent_pos": agent_pos}
