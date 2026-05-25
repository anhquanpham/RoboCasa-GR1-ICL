import datetime, uuid
from copy import deepcopy
from pathlib import Path
import gymnasium as gym
import numpy as np
import os
import robocasa  # we need this to register environments  # noqa: F401
import robosuite
from gymnasium import spaces
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.models.robots import (
    GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    gather_robot_observations,
    make_key_converter,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


def create_env_robosuite(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    controller_configs=None,
    camera_names=[
        "egoview",
        "robot0_eye_in_left_hand",
        "robot0_eye_in_right_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    enable_render=True,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    if controller_configs is None:
        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
        seed=seed,
        translucent_robot=False,
    )
    env_class = REGISTERED_ENVS[env_name]

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


class RoboCasaEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        robots_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        dump_rollout_dataset_dir=None,
        render_camera_names=None,
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = make_key_converter(robots_name)
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        # Append render-only cameras (not used by policy, only for video composite)
        self.render_camera_names = list(render_camera_names) if render_camera_names else []
        all_camera_names = list(camera_names) + self.render_camera_names
        all_camera_widths = [camera_widths] * len(camera_names) + [camera_widths] * len(self.render_camera_names)
        all_camera_heights = [camera_heights] * len(camera_names) + [camera_heights] * len(self.render_camera_names)

        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots_name.split("_")[0],
        )
        if (
            robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY
        ):
            controller_configs["type"] = "BASIC"
            controller_configs["composite_controller_specific_configs"] = {}
            controller_configs["control_delta"] = False

        self.env, self.env_kwargs = create_env_robosuite(
            env_name=env_name,
            robots=robots_name.split("_"),
            controller_configs=controller_configs,
            camera_names=all_camera_names,
            camera_widths=all_camera_widths,
            camera_heights=all_camera_heights,
            enable_render=enable_render,
            **kwargs,  # Forward kwargs to create_env_robosuite
        )

        # TODO: the following info should be output by grootrobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None
        self.render_extra_cache = []

        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            min_value, max_value = -1, 1
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[obs_name] = this_space

        for camera_name in camera_names:
            shape = [camera_heights, camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.groot_exporter = None
        self.np_exporter = None
        self.guidance_traj_visible_to_cameras = False
        self._guidance_traj_base = None
        self._guidance_markers_initialized = False

    def set_guidance_trajectory(self, trajectory_path: str | None):
        if trajectory_path is None:
            self._guidance_traj_base = None
            self._guidance_markers_initialized = False
            return
        p = Path(trajectory_path)
        if not p.exists():
            raise FileNotFoundError(f"Guidance trajectory file not found: {p}")
        arr = np.load(str(p), allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "states" in arr:
                arr = arr["states"]
            else:
                arr = arr[list(arr.keys())[0]]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Invalid guidance trajectory shape: {arr.shape} from {p}")
        self._guidance_traj_base = arr[:, :3]
        self._guidance_markers_initialized = False

    def _mobilebase_support_pose(self):
        sim = self.env.sim
        support_bid = sim.model.body_name2id("mobilebase0_support")
        support_pos = sim.data.body_xpos[support_bid].astype(np.float32)
        support_rot = sim.data.body_xmat[support_bid].reshape(3, 3).astype(np.float32)
        return support_pos, support_rot

    def _right_eef_in_mobilebase_support(self):
        sim = self.env.sim
        support_pos, support_rot = self._mobilebase_support_pose()
        robot = self.env.robots[0]
        eef_name = robot.robot_model.eef_name["right"]
        eef_pos = sim.data.get_body_xpos(eef_name).astype(np.float32)
        return (support_rot.T @ (eef_pos - support_pos)).astype(np.float32)

    def _set_mobilebase_support_marker_pos(self, name: str, pos_base: np.ndarray):
        body_id = self.env.sim.model.body_name2id(name)
        self.env.sim.model.body_pos[body_id] = np.asarray(pos_base, dtype=np.float32)

    def _update_guidance_markers(self, raw_obs: dict):
        if not self.guidance_traj_visible_to_cameras or self._guidance_traj_base is None:
            return
        sim = self.env.sim
        max_markers = 128
        n = min(len(self._guidance_traj_base), max_markers)

        for i in range(max_markers):
            name = f"guidance_traj_marker_{i:03d}"
            if i < n:
                self._set_mobilebase_support_marker_pos(name, self._guidance_traj_base[i])
            else:
                self._set_mobilebase_support_marker_pos(name, np.array([0.0, 0.0, -10.0], dtype=np.float32))

        self._set_mobilebase_support_marker_pos("guidance_current_waypoint_marker", self._guidance_traj_base[0])
        self._set_mobilebase_support_marker_pos(
            "guidance_right_eef_marker", self._right_eef_in_mobilebase_support()
        )
        sim.forward()
        self._guidance_markers_initialized = True

    def get_basic_observation(self, raw_obs):
        raw_obs.update(gather_robot_observations(self.env))
        self._update_guidance_markers(raw_obs)

        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        if self.render_camera_names:
            self.render_extra_cache = [raw_obs[f"{name}_image"] for name in self.render_camera_names]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        raw_obs = self.env.reset()
        # return obs
        obs = self.get_basic_observation(raw_obs)

        info = {}
        info["success"] = False
        info["grasp_distractor_obj"] = False

        return obs, info

    def step(self, action_dict):
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        env_action = np.concatenate(env_action)

        raw_obs, reward, done, info = self.env.step(env_action)

        obs = self.get_basic_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0
        info["grasp_distractor_obj"] = False
        if hasattr(self, "_check_grasp_distractor_obj"):
            info["grasp_distractor_obj"] = self._check_grasp_distractor_obj()

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        if not self.render_extra_cache:
            return self.render_cache
        # 2×2 grid layout when exactly 3 extra cameras are provided:
        #   top-left: render_cache (ego)  | top-right: extra[0] (lookback)
        #   bot-left: extra[1] (warp_left)| bot-right: extra[2] (warp_right)
        if len(self.render_extra_cache) == 3:
            tl = self.render_cache
            tr = self.render_extra_cache[0]
            bl = self.render_extra_cache[1]
            br = self.render_extra_cache[2]
            top = np.concatenate([tl, tr], axis=1)
            bot = np.concatenate([bl, br], axis=1)
            return np.concatenate([top, bot], axis=0)
        # Fallback: horizontal concat for any other number of extra cameras
        frames = [self.render_cache] + self.render_extra_cache
        return np.concatenate(frames, axis=1)

    def close(self):
        self.env.close()
