from gym import core, spaces
from dm_control import suite
from dm_env import specs
import numpy as np
from ipdb import set_trace


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int64(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        is_distracting_cs=None,
        distracting_cs_intensity=None,
        background_dataset_paths=None,
        environment_kwargs=None,
        channels_first=True,
    ):
        assert (
            "random" in task_kwargs
        ), "please specify a seed, for deterministic behaviour"
        self._domain_name = domain_name
        self._task_name = task_name
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._is_distracting_cs = is_distracting_cs
        self._distracting_cs_intensity = distracting_cs_intensity
        self._background_dataset_paths = background_dataset_paths
        self._channels_first = channels_first

        # create task
        if is_distracting_cs:
            from env.distracting_control import suite as dc_suite

            self._env = dc_suite.load(
                domain_name,
                task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
                difficulty=distracting_cs_intensity,
                dynamic=True,
                background_dataset_paths=background_dataset_paths,
            )
        else:
            from dm_control import suite as dm_suite

            self._env = dm_suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
            )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Dict()
            self._observation_space["visual"] = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
            if not is_distracting_cs:
                self._observation_space["state"] = _spec_to_box(
                    self._env.observation_spec().values(), np.float64
                )
            else:
                state_spec = []
                for key in self._env.observation_spec().keys():
                    if key != "visual":
                        state_spec.append(self._env.observation_spec()[key])
                self._observation_space["state"] = _spec_to_box(state_spec, np.float64)
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(), np.float64
            )

        self._state_space = (
            self._observation_space
            if not from_pixels
            else self._observation_space["state"]
        )

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get("random", 1))

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = {}
            obs["visual"] = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                obs["visual"] = obs["visual"].transpose(2, 0, 1).copy()
            if self._is_distracting_cs:
                obs_pieces = []
                for key in time_step.observation.keys():
                    if key == "visual":
                        continue
                    v = time_step.observation[key]
                    flat = np.array([v]) if np.isscalar(v) else v.ravel()
                    obs_pieces.append(flat)
                obs["state"] = np.concatenate(obs_pieces, axis=0)
            else:
                obs["state"] = _flatten_obs(time_step.observation)
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, done, False, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
