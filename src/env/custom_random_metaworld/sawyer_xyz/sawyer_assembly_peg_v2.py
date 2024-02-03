import mujoco
import numpy as np
from gymnasium.spaces import Box
from ..utils import quaternion_multiply

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)
from ipdb import set_trace


class SawyerNutAssemblyEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self, tasks=None, render_mode=None, random_level=1):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        self.random_level = random_level
        if random_level == 1:
            obj_low = (0, 0.6, 0.02)
            obj_high = (0, 0.6, 0.02)
            goal_low = (-0.1, 0.75, 0.1)
            goal_high = (0.1, 0.8, 0.1)
        if random_level == 2:
            obj_low = (-0.1, 0.55, 0.02)
            obj_high = (0.1, 0.6, 0.02)
            goal_low = (-0.2, 0.7, 0.1)
            goal_high = (0.2, 0.8, 0.1)
        if random_level >= 3:
            obj_low = (-0.4, 0.5, 0.02)
            obj_high = (0.4, 0.6, 0.02)
            goal_low = (-0.3, 0.7, 0.1)
            goal_high = (0.3, 0.8, 0.1)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
        )
        if tasks is not None:
            self.tasks = tasks

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }

        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        if random_level <= 3:
            self._random_reset_space = Box(
                np.hstack((obj_low, goal_low)),
                np.hstack((obj_high, goal_high)),
            )
        else:
            self._random_reset_space = Box(
                np.hstack((obj_low, goal_low, (-0.15,))),
                np.hstack((obj_high, goal_high, (0.15,))),
            )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_assembly_peg.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(success),
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [("pegTop", self._target_pos)]

    def _get_id_main_object(self):
        """TODO: Reggie"""
        return self.unwrapped.model.geom_name2id("WrenchHandle")

    def _get_pos_objects(self):
        return self.data.site_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "RoundNut-8")
        ]

    def _get_quat_objects(self):
        return self.data.body("RoundNut").xquat

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict["state_achieved_goal"] = self.get_body_com("RoundNut")
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        z = 0.0
        goal_pos = self._get_state_rand_vec()
        if self.random_level == 4:
            z = goal_pos[-1]
            goal_pos = goal_pos[:-1]
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
            goal_pos = self._get_state_rand_vec()
            if self.random_level == 4:
                z = goal_pos[-1]
                goal_pos = goal_pos[:-1]
        w = (1 - z ** 2) ** 0.5
        quaternion = np.array([w, 0.0, 0.0, z])
        quaternion_ideal = np.array([0.707, 0, 0, 0.707])
        quaternion = quaternion_multiply(quaternion, quaternion_ideal)
        self.obj_init_pos = goal_pos[:3]
        self._target_pos = goal_pos[-3:]
        peg_pos = self._target_pos - np.array([0.0, 0.0, 0.05])

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        ] = peg_pos
        self.model.site_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pegTop")
        ] = self._target_pos
        self.data.qpos[12:] = quaternion
        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = aligned and hooked

        # Target height is a 3D funnel centered on the peg.
        # use the success flag to widen the bottleneck once the agent
        # learns to place the wrench on the peg -- no reason to encourage
        # tons of alignment accuracy if task is already solved
        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1.0, 1.0, 3.0])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * scale),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid="long_tail",
        )

        return in_place, success

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos("RoundNut")
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = SawyerNutAssemblyEnvV2.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = SawyerNutAssemblyEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions,
            wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            medium_density=True,
        )
        reward_in_place, success = SawyerNutAssemblyEnvV2._reward_pos(
            wrench_center, self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success
        if success:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )


class TrainAssemblyv2(SawyerNutAssemblyEnvV2):
    tasks = None

    def __init__(self):
        SawyerNutAssemblyEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)


class TestAssemblyv2(SawyerNutAssemblyEnvV2):
    tasks = None

    def __init__(self):
        SawyerNutAssemblyEnvV2.__init__(self, self.tasks)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
