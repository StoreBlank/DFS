from collections import deque
import utils
import gymnasium as gym
import gymnasium.spaces as spaces
import cv2
import numpy as np
from einops import rearrange
from ipdb import set_trace


class MetaworldWrapper(gym.Wrapper):
    def __init__(self, env, mode='rgb', image_size=400):
        assert mode in ['rgb', 'rgbd']
        super().__init__(env)
        self.mode = mode
        self.image_size = image_size
        self._observation_space = self.init_observation_space(env.observation_space, mode=mode, image_size=image_size)

    @staticmethod
    def init_observation_space(state_space, mode='rgb', image_size=400):
        if mode == 'rgb':
            visual_space = spaces.Box(0, np.inf, shape=(3, image_size, image_size))
        else:
            visual_space = spaces.Box(0, np.inf, shape=(4, image_size, image_size))

        return spaces.Dict({"visual": visual_space, "state": state_space})

    def convert_observation(self, observation):
        image = np.array(self.env.mujoco_renderer.render('rgb_array', camera_id=2))
        if self.mode == 'rgbd':
            image = np.concatenate([image, self.mujoco_renderer.render('depth_array', camera_id=2)], axis=-1)

        # resize image
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = rearrange(image, "h w c -> c h w")

        return dict(visual=image, state=observation)

    def reset(self):
        observation, info = self.env.reset()
        return self.convert_observation(observation), info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        return self.convert_observation(observation), reward, done, truncated, info
    
    def render(self, mode='rgb_array', height=448, width=448, camera_id=2):
        if mode != 'rgb_array':
            raise ValueError(f"Unsupported render mode: {mode}. Only 'rgb_array' is supported.")

        image = self.mujoco_renderer.render(render_mode=mode, camera_id=camera_id)
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_CUBIC)
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""
    def __init__(self, env, k, done_on_success):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._done_on_success = done_on_success
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
        if info['success'] and self._done_on_success:
            done = True
        self._frames.append(obs['visual'])
        new_obs = {
            'visual': self._get_visual_obs(),
            'state': obs['state']
        }
        return new_obs, reward, done, info
    
    def render(self, mode='rgb_array', height=448, width=448, camera_id=2):
        return self.env.render(mode=mode, height=height, width=width, camera_id=camera_id)

    def _get_visual_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


def wrap(env, frame_stack=3, mode='rgb', image_size=400, done_on_success=True):
    env = MetaworldWrapper(env, mode=mode, image_size=image_size)
    env = FrameStack(env, frame_stack, done_on_success)
    return env
