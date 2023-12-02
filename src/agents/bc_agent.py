import numpy as np
import torch

import utils
import agents.modules as m
from agents.sac_agent import SAC, StateSAC

def kl_divergence(mu1, log_std1, mu2, log_std2):
    """KL divergence between two Gaussians."""
    k = mu1.size(1)
    std1, std2 = log_std1.exp(), log_std2.exp()
    residual = log_std2 - log_std1 + (mu1 - mu2) ** 2 / std2 + std1 / std2
    residual = residual.sum(1)
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
            L.log("train_student/actor_loss", loss, step)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_aug_sample()
        else:
            obs, mu_target, log_std_target, _, _, _ = replay_buffer.behavior_sample()

        self.update_actor(obs, mu_target, log_std_target, L, step)
