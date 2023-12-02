import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import utils
import agents.modules as m
import ipdb


class SAC(object):
    def __init__(self, obs_shape, action_shape, args, actor=None, critic=None):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq

        self.actor = actor.cuda()
        self.critic = critic.cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        # must rewrite the function in new agent to get obs["state"] or obs["visual"]
        _obs = torch.FloatTensor(obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def exhibit_behavior(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, log_std = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten(), log_std.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train/critic_loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log("train/actor_loss", actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train/alpha_loss", alpha_loss, step)
                L.log("train/alpha_value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def freeze(self):
        for param in self.actor.parameters():
            param.requires_grad = False

        for param in self.critic.parameters():
            param.requires_grad = False

        self.log_alpha.requires_grad = False


class StateSAC(SAC):
    def __init__(self, obs_shape, action_shape, args):
        actor = m.StateActor(obs_shape, action_shape, args.hidden_dim)
        critic = m.StateCritic(obs_shape, action_shape, args.hidden_dim)
        super().__init__(obs_shape, action_shape, args, actor, critic)

    def _obs_to_input(self, obs):
        _obs = obs["state"]
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = obs["state"]
        next_obs = next_obs["state"]
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()


class VisualSAC(SAC):
    def __init__(self, obs_shape, action_shape, args):
        shared_cnn = m.SharedCNN(
            obs_shape, args.num_shared_layers, args.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, args.num_head_layers, args.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        self.use_aug = args.use_aug

        actor = m.VisualActor(actor_encoder, action_shape, args.hidden_dim)
        critic = m.VisualCritic(critic_encoder, action_shape, args.hidden_dim)
        super().__init__(obs_shape, action_shape, args, actor, critic)

    def _obs_to_input(self, obs):
        _obs = obs["visual"]
        if isinstance(_obs, utils.LazyFrames):
            _obs = np.array(_obs)
        else:
            _obs = _obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log("train/actor_loss", actor_loss, step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train/alpha_loss", alpha_loss, step)
                L.log("train/alpha_value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, action, reward, next_obs, not_done = replay_buffer.aug_sample()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
        obs = obs["visual"]
        next_obs = next_obs["visual"]
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
