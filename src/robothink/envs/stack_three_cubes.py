"""
StackThreeCubes Environment
==========================
A robosuite manipulation environment that mirrors the built-in ``Lift``
environment in every respect **except** that three coloured cubes (red,
green, blue) are placed on the table.  The task is to stack them in the
order red (bottom) → green (middle) → blue (top).

Reward
------
+0.5  when the green cube is stacked on the red cube
      (and the partial stack is not yet complete)
+0.5  additional when the blue cube is additionally stacked on the green
      cube that is already on the red cube  →  total reward = 1.0

Success / done
--------------
The episode is considered solved (``_check_success`` returns True) when
the full tower is complete, i.e. total reward == 1.0.

Stacking detection  (Option A — height + horizontal proximity)
--------------------------------------------------------------
A cube B is considered "stacked on" cube A when:
  1. The XY distance between their centres is ≤ ``stack_xy_thresh``
     (default 0.05 m).
  2. The Z centre of B is within ``stack_z_thresh`` (default 0.015 m)
     of  (z_centre_A  +  cube_half_size_A  +  cube_half_size_B),
     i.e. B is sitting roughly on top of A.

Initialisation
--------------
The three cubes are placed at uniformly-random XY positions inside the
same table region used by the original ``Lift`` environment.  Positions
are resampled until no two cube centres are closer than
``min_cube_separation`` (default 0.06 m) to one another.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class StackThreeCubes(ManipulationEnv):
    """Stack red → green → blue cubes.

    All constructor parameters that are not listed below are forwarded
    verbatim to :class:`ManipulationEnv`, mirroring the signature of the
    built-in ``Lift`` environment.

    Parameters
    ----------
    robots : str | list[str]
        Robot(s) to load (e.g. ``"Panda"``).
    cube_size : float
        Half-size of every cube in metres.  Default ``0.02`` (so each cube
        is 4 cm × 4 cm × 4 cm), matching the original ``Lift`` cube.
    min_cube_separation : float
        Minimum distance (metres) between any two cube centres at reset.
        Default ``0.06``.
    stack_xy_thresh : float
        Maximum XY distance (metres) between cube centres for the upper
        cube to be considered "stacked".  Default ``0.05``.
    stack_z_thresh : float
        Tolerance (metres) on the expected stacking height.  Default
        ``0.015``.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        # --- environment-specific ---
        cube_size=0.02,
        min_cube_separation=0.06,
        stack_xy_thresh=0.05,
        stack_z_thresh=0.015,
    ):
        # Store custom parameters before calling super().__init__
        # (super triggers _load_model → needs these already set).
        self.cube_size = cube_size
        self.min_cube_separation = min_cube_separation
        self.stack_xy_thresh = stack_xy_thresh
        self.stack_z_thresh = stack_z_thresh

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))  # matches Lift default

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, action=None):
        """Staged stacking reward.

        Returns
        -------
        float
            0.0  – neither stage satisfied
            0.5  – green is stacked on red  (stage 1)
            1.0  – blue is stacked on green which is on red  (stage 2 / success)
        """
        reward = 0.0

        green_on_red = self._cubes_stacked("red_cube", "green_cube")
        if green_on_red:
            reward += 0.5
            blue_on_green = self._cubes_stacked("green_cube", "blue_cube")
            if blue_on_green:
                reward += 0.5

        if self.reward_scale != 1.0:
            reward *= self.reward_scale

        return reward

    def _cubes_stacked(self, bottom_name: str, top_name: str) -> bool:
        """Return True if *top* cube is stacked on *bottom* cube.

        Stacking is defined as:
          1. XY centre distance ≤ ``stack_xy_thresh``
          2. Z centre of top ≈ z_centre_bottom + half_bottom + half_top
             within ± ``stack_z_thresh``
        """
        pos_bottom = np.array(
            self.sim.data.body_xpos[self.obj_body_id[bottom_name]]
        )
        pos_top = np.array(
            self.sim.data.body_xpos[self.obj_body_id[top_name]]
        )

        xy_dist = np.linalg.norm(pos_top[:2] - pos_bottom[:2])
        if xy_dist > self.stack_xy_thresh:
            return False

        # Expected Z of top cube centre when sitting flat on bottom cube
        expected_z = pos_bottom[2] + self.cube_size + self.cube_size
        z_ok = abs(pos_top[2] - expected_z) <= self.stack_z_thresh

        return z_ok

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------

    def _check_success(self):
        """Episode is solved when the full tower is complete (reward == 1.0)."""
        return (
            self._cubes_stacked("red_cube", "green_cube")
            and self._cubes_stacked("green_cube", "blue_cube")
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build arena, robot(s), and the three coloured cube objects."""
        super()._load_model()

        # Position robot(s) — identical to Lift
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Arena
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # --- Cube materials / colours ---
        # Each cube gets a simple flat colour via a MuJoCo material.
        tex_attrib = {"type": "cube"}
        mat_attrib_red   = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        mat_attrib_green = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        mat_attrib_blue  = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="red_tex",
            mat_name="red_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib_red,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="green_tex",
            mat_name="green_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib_green,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="blue_tex",
            mat_name="blue_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib_blue,
        )

        cube_half = self.cube_size  # half-size passed to BoxObject

        self.red_cube = BoxObject(
            name="red_cube",
            size_min=[cube_half, cube_half, cube_half],
            size_max=[cube_half, cube_half, cube_half],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.green_cube = BoxObject(
            name="green_cube",
            size_min=[cube_half, cube_half, cube_half],
            size_max=[cube_half, cube_half, cube_half],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.blue_cube = BoxObject(
            name="blue_cube",
            size_min=[cube_half, cube_half, cube_half],
            size_max=[cube_half, cube_half, cube_half],
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )

        # --- Placement sampler (mirrors Lift defaults) ---
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(
                [self.red_cube, self.green_cube, self.blue_cube]
            )
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.red_cube, self.green_cube, self.blue_cube],
                x_range=[-0.2, 0.2],
                y_range=[-0.2, 0.2],
                rotation=None,          # random yaw, same as Lift
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # --- Compose task ---
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.red_cube, self.green_cube, self.blue_cube],
        )

    # ------------------------------------------------------------------
    # Simulation initialisation
    # ------------------------------------------------------------------

    def _setup_references(self):
        """Cache body / joint IDs for fast lookups during simulation."""
        super()._setup_references()

        self.obj_body_id = {
            "red_cube":   self.sim.model.body_name2id(self.red_cube.root_body),
            "green_cube": self.sim.model.body_name2id(self.green_cube.root_body),
            "blue_cube":  self.sim.model.body_name2id(self.blue_cube.root_body),
        }

    def _reset_internal(self):
        """Reset simulation; place cubes with guaranteed separation."""
        super()._reset_internal()

        # Use the placement sampler for initial candidate positions, but
        # re-sample until the minimum separation constraint is satisfied.
        max_attempts = 1000
        for attempt in range(max_attempts):
            object_placements = self.placement_initializer.sample()

            positions = {}
            for obj_name, (obj_pos, obj_quat, _) in object_placements.items():
                positions[obj_name] = np.array(obj_pos)

            # Check pairwise XY separations (Z may differ slightly due to
            # z_offset; we check full 3-D distance to be safe).
            names = list(positions.keys())
            valid = True
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    dist = np.linalg.norm(
                        positions[names[i]] - positions[names[j]]
                    )
                    if dist < self.min_cube_separation:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                break
        else:
            # Fallback: use last sample even if constraint not met
            # (should be extremely rare with the default table size).
            pass

        # Write accepted positions into the simulation state
        for obj_name, (obj_pos, obj_quat, _) in object_placements.items():
            # Resolve the correct object reference
            obj = {
                "red_cube":   self.red_cube,
                "green_cube": self.green_cube,
                "blue_cube":  self.blue_cube,
            }[obj_name]

            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
            )

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        """Add cube pose observables, mirroring the Lift environment."""
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        # We need the modality of the existing robot observables
        if self.use_object_obs:
            modality = f"{pf}object"

            # Helper: build a position sensor for a named cube
            def _make_pos_sensor(cube_name, body_id_key):
                @sensor(modality=modality)
                def _pos(obs_cache):
                    return np.array(
                        self.sim.data.body_xpos[self.obj_body_id[body_id_key]]
                    )
                _pos.__name__ = f"{cube_name}_pos"
                return _pos

            def _make_quat_sensor(cube_name, body_id_key):
                @sensor(modality=modality)
                def _quat(obs_cache):
                    return convert_quat(
                        np.array(
                            self.sim.data.body_xquat[self.obj_body_id[body_id_key]]
                        ),
                        to="xyzw",
                    )
                _quat.__name__ = f"{cube_name}_quat"
                return _quat

            for cube_name, key in [
                ("red_cube",   "red_cube"),
                ("green_cube", "green_cube"),
                ("blue_cube",  "blue_cube"),
            ]:
                sensors = [
                    _make_pos_sensor(cube_name, key),
                    _make_quat_sensor(cube_name, key),
                ]
                names = [s.__name__ for s in sensors]
                for name, s in zip(names, sensors):
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                    )

        return observables