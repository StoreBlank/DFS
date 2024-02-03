import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import agents.modules as m

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange
from ipdb import set_trace


class DiffusionAgent(object):
    def __init__(self, agent_config):
        self.shape_meta = agent_config.shape_meta
        self.horizon = agent_config.horizon
        self.n_obs_steps = agent_config.n_obs_steps
        # assert agent_config.n_action_steps == 1, "multi-step not implemented yet"
        self.n_action_steps = agent_config.n_action_steps

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
        )
        rgb_model = get_resnet('resnet18')
        obs_encoder = MultiImageObsEncoder(
            self.shape_meta,
            rgb_model,
            crop_shape=(agent_config.crop_size, agent_config.crop_size),
        )
        self.model = DiffusionUnetImagePolicy(
            self.shape_meta,
            noise_scheduler,
            obs_encoder,
            self.horizon,
            self.n_action_steps,
            self.n_obs_steps,
        ).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=agent_config.lr
        )
        # 0-255 -> 0-1
        self.pre_normalizer = m.NormalizeImg().cuda()

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def eval(self):
        self.train(False)

    # inference
    def _obs_to_input(self, obs):
        _obs = obs['visual']
        if isinstance(_obs, utils.LazyFrames):
            _obs = np.array(_obs)
        else:
            _obs = _obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = self.pre_normalizer(_obs)
        _obs = rearrange(_obs, '(t c) h w -> t c h w', t=self.n_obs_steps).unsqueeze(0)
        return { 'visual': _obs }

    def select_action(self, obs):
        # only support single action step
        _obs = self._obs_to_input(obs)
        actions = self.model.predict_action(_obs)['action']
        action = actions.squeeze(0).cpu().data.numpy().flatten()
        return action

    def select_actions(self, obs):
        _obs = self._obs_to_input(obs)
        actions = self.model.predict_action(_obs)['action']
        actions = actions.squeeze(0).cpu().data.numpy()
        return actions

    # training
    def set_normalizer(self, normalizer):
        self.model.set_normalizer(normalizer.cuda())
        
    def update(self, diffusion_replay_buffer, L, step):
        obs_dict = diffusion_replay_buffer.sample()
        obs_dict['obs']['visual'] = self.pre_normalizer(obs_dict['obs']['visual'])

        loss = self.model.compute_loss(obs_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if L is not None:
            L.log('train/loss', loss, step)
