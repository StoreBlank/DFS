from collections import deque
import utils
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
from einops import rearrange
import mani_skill2.envs
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from ipdb import set_trace


class ManiSkillWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = self.init_observation_space(env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        # States include robot proprioception (agent) and task information (extra)
        state_spaces = []
        state_spaces.extend(flatten_dict_space_keys(obs_space["agent"]).spaces.values())
        state_spaces.extend(flatten_dict_space_keys(obs_space["extra"]).spaces.values())
        # Concatenate all the state spaces
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-np.inf, np.inf, shape=(state_size,))

        # Concatenate all the image spaces
        image_shapes = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            image_shapes.append(cam_space["rgb"].shape)
            image_shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(image_shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(c, h, w))

        # Create the new observation space
        return spaces.Dict({"visual": rgbd_space, "state": state_space})

    @staticmethod
    def convert_observation(observation):
        images = []
        for cam_uid, cam_obs in observation["image"].items():
            rgb = cam_obs["rgb"]
            depth = cam_obs["depth"]

            if isinstance(rgb, torch.Tensor):
                rgb = rgb.to(device="cpu", non_blocking=True).numpy()
            if isinstance(depth, torch.Tensor):
                depth = depth.to(device="cpu", non_blocking=True).numpy()

            images.append(rgb)
            images.append(depth)

        # Concatenate all the images
        rgbd = np.concatenate(images, axis=-1)
        rgbd = rearrange(rgbd, "h w c -> c h w")

        # Concatenate all the states
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        return dict(visual=rgbd, state=state)
    
    def observation(self, observation):
        return self.convert_observation(observation)


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space['visual'].shape
        self.observation_space = spaces.Dict()
        self.observation_space['visual'] = spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=np.uint8,
        )
        self.observation_space['state'] = env.observation_space['state']

    def reset(self):
        obs, _ = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs['visual'])
        return {
            'visual': self._get_visual_obs(),
            'state': obs['state']
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._frames.append(obs['visual'])
        new_obs = {
            'visual': self._get_visual_obs(),
            'state': obs['state']
        }
        return new_obs, reward, done, info

    def _get_visual_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


def make_env(
        env_id,
        frame_stack=3,
        reward_mode=None,
        control_mode=None,
        render_mode=None,
        renderer_kwargs=None,
    ):
        env = gym.make(
            env_id,
            obs_mode="rgbd",
            reward_mode=reward_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            renderer_kwargs=renderer_kwargs,
        )
        env = ManiSkillWrapper(env)
        env = FrameStack(env, frame_stack)
        return env
