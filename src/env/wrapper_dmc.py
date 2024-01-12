import os, sys
import gym
import utils
import dmc2gym
import numpy as np
from collections import deque

from ipdb import set_trace

def make_env(
		domain_name,
		task_name,
		seed=0,
		episode_length=1000,
		frame_stack=3,
		action_repeat=4,
		image_size=100,
		mode='train',
		intensity=0.,
		mask_model = None,
	):
	"""Make environment for experiments"""
	assert mode in {'train', 'distracting_cs'}, \
		f'specified mode "{mode}" is not supported'

	paths = []
	is_distracting_cs = (mode == 'distracting_cs' and intensity > 0.)
	if is_distracting_cs:
		import env.distracting_control.suite as dc_suite
		loaded_paths = [os.path.join(dir_path, 'DAVIS/JPEGImages/480p') for dir_path in utils.load_config('datasets')]
		for path in loaded_paths:
			if os.path.exists(path):
				paths.append(path)
	env = dmc2gym.make(
		domain_name=domain_name,
		task_name=task_name,
		seed=seed,
		visualize_reward=False,
		from_pixels=True,
		height=image_size,
		width=image_size,
		episode_length=episode_length,
		frame_skip=action_repeat,
		is_distracting_cs=is_distracting_cs,
		distracting_cs_intensity=intensity,
		background_dataset_paths=paths
	)
	env = FrameStack(env, frame_stack, mask_model)

	return env


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k, mask_model=None):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		self.mask_model = mask_model
		shp = env.observation_space['visual'].shape
		self.observation_space = gym.spaces.Dict()
		self.observation_space['visual'] = gym.spaces.Box(
			low=0,
			high=255,
			shape=((shp[0] * k,) + shp[1:]),
			dtype=np.uint8,
		)
		self.observation_space['state'] = env.observation_space['state']
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset() # obs["visual"]: numpy array [3, 100, 100]
		if self.mask_model:
			mask = self.mask_model.get_mask(obs["visual"])
			obs["visual"] = (mask*obs["visual"]).numpy().astype(np.uint8)
		for _ in range(self._k):
			self._frames.append(obs['visual'])
		return {
			'visual': self._get_visual_obs(),
			'state': obs['state']
		}

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action) # obs["visual"]: numpy array [3, 100, 100] uint8
		# from PIL import Image
		# img = np.transpose(obs["visual"], (1,2,0))
		# print(img.shape)
		# img = Image.fromarray(img)
		# img.save("./obs.jpg")
		if self.mask_model:
			mask = self.mask_model.get_mask(obs["visual"]) # tensor int8
			obs["visual"] = (mask*obs["visual"]).numpy().astype(np.uint8)
			# """
			# test code
			# """
			# from PIL import Image
			# img = np.transpose(obs["visual"], (1,2,0))
			# img = Image.fromarray(img)
			# img.save("./obs_masked.jpg")

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

"""
obs -> mask_func(obs) -> masked_obs: derectly usable in RL algorithm

mask_func(obs, cfg.attn_map_extractor, cfg.map_type, cfg.adapter): -> masked_obs
	obs: original obs stack in env

	cfg.attn_map_extractor: KeypointDetector / TextualInversionedCrossAttn
	cfg.map_type: HardMask / ProbMask
	cfg.adapter: use adapter for Gaussian prob map, textual inversion, no adapter
"""