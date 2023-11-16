import torch
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

        self.actor = m.VisualActor(actor_encoder, action_shape, args.hidden_dim).cuda()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.bc_lr, betas=(args.actor_beta, 0.999)
        )

        self.expert = None

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def set_expert(self, expert: StateSAC):
        self.expert = expert
        self.expert.freeze()

    def update_actor(self, obs, L=None, step=None):
        if self.expert is None:
            raise Exception("Expert not set")
        obs_state = obs["state"]
        obs_visual = obs["visual"]

        with torch.no_grad():
            mu_target, _, _, log_std_target = self.expert.actor(obs_state, False, False)
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
        obs, _, _, _, _ = replay_buffer.sample()

        self.update_actor(obs, L, step)
