import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import agents.modules as m
from agents.sac_agent import SAC, StateSAC
from ipdb import set_trace


eps = 1e-7


def kl_divergence(mu1, log_std1, mu2, log_std2):
    """KL divergence between two Gaussians."""
    k = mu1.size(1)
    std1, std2 = log_std1.exp(), log_std2.exp()
    residual = log_std2 - log_std1 + (mu1 - mu2) ** 2 / (std2 + eps) + std1 / (std2 + eps)
    residual = residual.sum(1)
    if residual.mean() == torch.inf:
        set_trace()
    return 0.5 * (residual - k)


class BC(SAC):
    def __init__(self, agent_obs_shape, action_shape, agent_config):
        shared_cnn = m.SharedCNN(
            agent_obs_shape, agent_config.num_shared_layers, agent_config.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, agent_config.num_head_layers, agent_config.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, agent_config.projection_dim),
        )
        self.use_aug = agent_config.use_aug

        self.actor = m.VisualActor(actor_encoder, action_shape, agent_config.hidden_dim).cuda()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_config.bc_lr, betas=(agent_config.actor_beta, 0.999)
        )

        self.train()

    # rewrite obs
    def _obs_to_input(self, obs):
        _obs = obs["visual"]
        if isinstance(_obs, utils.LazyFrames):
            _obs = np.array(_obs)
        else:
            _obs = _obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def update_actor(self, obs, mu_target, log_std_target, L=None, step=None):
        obs_visual = obs["visual"]

        mu_pred, _, _, log_std_pred = self.actor(obs_visual, False, False)

        # Mention the squashing
        loss = kl_divergence(
            torch.atanh(mu_pred), log_std_pred, torch.atanh(mu_target), log_std_target
        ).mean()

        if L is not None:
            L.log("train_bc/actor_loss", loss, step)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, _, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_aug_sample()
        else:
            obs, _, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_sample()

        self.update_actor(obs, mu_target, log_std_target, L, step)


class FeatBaselineBC(BC):
    def __init__(self, agent_obs_shape, action_shape, agent_config):
        super().__init__(agent_obs_shape, action_shape, agent_config)
        self.expert = None
        self.criterion = m.CRDLoss(agent_config).cuda()

        self.crd_optimizer = torch.optim.Adam(
            self.criterion.parameters(), lr=agent_config.crd_baseline_lr, betas=(agent_config.actor_beta, 0.999)
        )

        self.train()

    @staticmethod
    def load_baseline(load_path, obs_shape, action_shape, agent_config):
        agent = torch.load(load_path)
        agent.train()
        baseline = FeatBaselineBC(obs_shape, action_shape, agent_config)
        baseline.actor = agent.actor
        return baseline

    def set_expert(self, expert):
        self.expert = expert
        self.expert.freeze()

    def clean_expert(self):
        self.expert = None

    def update_crd_baseline(self, replay_buffer, L, step):
        if self.use_aug:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_aug_sample(return_idxs=True)
        else:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_sample(return_idxs=True)

        obs_visual = obs['visual']
        obs_state = obs['state']

        with torch.no_grad():
            _, _, _, _, feats_t = self.expert.actor(obs_state, False, False, True, False)
            feat_t = feats_t[-1]
            _, _, _, _, feats_s = self.actor(obs_visual, False, False, False, True, False)
            feat_s = feats_s[-1]

        # debug
        # if step == 500:
        #     set_trace()
        loss_crd = self.criterion(feat_s, feat_t, idxs, replay_buffer)

        if L is not None:
            L.log('train_baseline/crd_loss', loss_crd, step)

        self.crd_optimizer.zero_grad()
        loss_crd.backward()
        self.crd_optimizer.step()

    def sample_feats(self, replay_buffer, n=1000):
        obs, _, _, _, _ = replay_buffer.sample(n)

        obs_visual = obs['visual']
        obs_state = obs['state']

        with torch.no_grad():
            _, _, _, _, feats_t = self.expert.actor(obs_state, False, False, True, False)
            feat_t = feats_t[-1]
            feat_t = self.criterion.embed_t(feat_t)
            feat_t = F.normalize(feat_t, dim=1)
            _, _, _, _, feats_s = self.actor(obs_visual, False, False, False, True, False)
            feat_s = feats_s[-1]
            feat_s = self.criterion.embed_s(feat_s)
            feat_s = F.normalize(feat_s, dim=1)

        # into numpy array
        feat_t = feat_t.cpu().detach().numpy()
        feat_s = feat_s.cpu().detach().numpy()

        return feat_s, feat_t


class CrdBC(BC):
    def __init__(self, agent_obs_shape, action_shape, agent_config):
        shared_cnn = m.SharedCNN(
            agent_obs_shape, agent_config.num_shared_layers, agent_config.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, agent_config.num_head_layers, agent_config.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, agent_config.projection_dim),
        )
        self.use_aug = agent_config.use_aug

        self.actor = m.VisualActor(actor_encoder, action_shape, agent_config.hidden_dim).cuda()

        self.expert = None
        self.criterion = m.CRDLoss(agent_config).cuda()

        self.optimizer = torch.optim.Adam(
            nn.ModuleList([self.actor, self.criterion]).parameters(),
            lr=agent_config.bc_lr,
            betas=(agent_config.actor_beta, 0.999),
        )
        self.lambda_crd = agent_config.lambda_crd

        self.train()

    def set_expert(self, expert):
        self.expert = expert
        self.expert.freeze()

    def clean_expert(self):
        self.expert = None

    def update_actor(self, obs, idxs, contrastive_buffer, L=None, step=None):
        obs_visual = obs['visual']
        obs_state = obs['state']

        mu_pred, _, _, log_std_pred, feats_s = self.actor(obs_visual, False, False, False, True, False)
        feat_s = feats_s[-1]
        with torch.no_grad():
            mu_target, _, _, log_std_target, feats_t = self.expert.actor(obs_state, False, False, True, False)
            feat_t = feats_t[-1]

        # loss_kl = kl_divergence(
        #     torch.atanh(mu_pred), log_std_pred, torch.atanh(mu_target), log_std_target
        # ).mean()
        loss_kl = (mu_pred - mu_target).pow(2).mean() + (log_std_pred - log_std_target).pow(2).mean()
        loss_crd = self.criterion(feat_s, feat_t, idxs, contrastive_buffer)
        loss = loss_kl + self.lambda_crd * loss_crd

        if L is not None:
            L.log('train_crd/actor_loss', loss, step)
            L.log('train_crd/kl_loss', loss_kl, step)
            L.log('train_crd/crd_loss', loss_crd, step)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_aug_sample(return_idxs=True)
        else:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_sample(return_idxs=True)

        self.update_actor(obs, idxs, replay_buffer, L, step)


class PureCrdBC(BC):
    def __init__(self, agent_obs_shape, action_shape, agent_config):
        shared_cnn = m.SharedCNN(
            agent_obs_shape, agent_config.num_shared_layers, agent_config.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, agent_config.num_head_layers, agent_config.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, agent_config.projection_dim),
        )
        self.use_aug = agent_config.use_aug

        self.actor = m.VisualActor(actor_encoder, action_shape, agent_config.hidden_dim).cuda()

        self.expert = None
        self.criterion = m.CRDLoss(agent_config).cuda()

        self.feature_optimizer = torch.optim.Adam(
            nn.ModuleList(self.actor.feature_layers() + [self.criterion]).parameters(),
            lr=agent_config.crd_lr,
            betas=(agent_config.actor_beta, 0.999),
        )
        self.last_layer_optimizer = torch.optim.Adam(
            nn.ModuleList(self.actor.last_layer()).parameters(),
            lr=agent_config.bc_lr,
            betas=(agent_config.actor_beta, 0.999),
        )

        self.train()

    def set_expert(self, expert):
        self.expert = expert
        self.expert.freeze()

    def clean_expert(self):
        self.expert = None

    def update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_aug_sample(return_idxs=True)
        else:
            obs, _, _, _, _, _, _, idxs = replay_buffer.behavior_sample(return_idxs=True)

        obs_visual = obs['visual']
        obs_state = obs['state']

        _, _, _, _, feats_s = self.actor(obs_visual, False, False, False, True, False)
        feat_s = feats_s[-1]
        with torch.no_grad():
            _, _, _, _, feats_t = self.expert.actor(obs_state, False, False, True, False)
            feat_t = feats_t[-1]

        loss = self.criterion(feat_s, feat_t, idxs, replay_buffer)

        if L is not None:
            L.log('train/pure_crd_crd_loss', loss, step)

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

    def sample_feats(self, replay_buffer, n=1000):
        obs, _, _, _, _ = replay_buffer.sample(n)

        obs_visual = obs['visual']
        obs_state = obs['state']

        with torch.no_grad():
            _, _, _, _, feats_t = self.expert.actor(obs_state, False, False, True, False)
            feat_t = feats_t[-1]
            feat_t = self.criterion.embed_t(feat_t)
            feat_t = F.normalize(feat_t, dim=1)
            _, _, _, _, feats_s = self.actor(obs_visual, False, False, False, True, False)
            feat_s = feats_s[-1]
            feat_s = self.criterion.embed_s(feat_s)
            feat_s = F.normalize(feat_s, dim=1)

        # into numpy array
        feat_t = feat_t.cpu().detach().numpy()
        feat_s = feat_s.cpu().detach().numpy()

        return feat_s, feat_t

    def update_last_layer(self, replay_buffer, L, step):
        if self.use_aug:
            obs, _, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_aug_sample()
        else:
            obs, _, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_sample()

        obs_visual = obs["visual"]
        # debug
        # set_trace()

        mu_pred, _, _, log_std_pred = self.actor(obs_visual, False, False)

        # Mention the squashing
        # loss = kl_divergence(
        #     torch.atanh(mu_pred).clamp(-10., 10.), log_std_pred, torch.atanh(mu_target).clamp(-10., 10.), log_std_target
        # ).clamp(0, 100.).mean()
        loss = (mu_pred - mu_target).pow(2).mean() + (log_std_pred - log_std_target).pow(2).mean()

        if L is not None:
            L.log("train/pure_crd_actor_loss", loss, step)

        self.last_layer_optimizer.zero_grad()
        loss.backward()
        # debug
        # set_trace()
        # for param in nn.ModuleList(self.actor.last_layer()).parameters():
        #     if (param.data.grad == torch.nan).any():
        #         param.data.grad[param.data.grad == torch.nan] = 0.
        self.last_layer_optimizer.step()
