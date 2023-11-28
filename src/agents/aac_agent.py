import numpy as np
import torch
import torch.nn.functional as F
import utils
import agents.modules as m

from agents.sac_agent import SAC

class AACAgent(SAC):
    def __init__(self, obs_shape, state_shape, action_shape, args):

        # visual actor
        if args.actor_weights is not None:
            actor_model = torch.load(args.actor_weights)
            actor=actor_model.actor
            print("--- actor loaded --- ")
        else:
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
            actor = m.VisualActor(actor_encoder, action_shape, args.actor_hidden_dim).cuda()
            print("--- actor inited ---")
        # actor optimizer
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.bc_lr, betas=(args.actor_beta, 0.999))

        # state critic
        if args.critic_weights is not None:
            critic_model = torch.load(args.critic_weights)
            critic = critic_model.critic
            print("--- critic loaded ---")
        else:
            critic = m.StateCritic(state_shape, action_shape, args.critic_hidden_dim)
            print("--- critic inited ---")
        # critic optimizer
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999))

        super().__init__(obs_shape, action_shape, args, actor, critic)
        self.use_aug=args.use_aug
        self.train() # set actor, critic trainable

    # rewrite obs 
    # NOTE: added mode to switch modal of observation
    def _obs_to_input(self, obs, mode):
        assert mode in ["visual", "state"], "Invalid mode, you must select from visual and state"
        if mode == "visual":
            _obs = obs["visual"]
            if isinstance(_obs, utils.LazyFrames):
                _obs = np.array(_obs)
            else:
                _obs = _obs
        else:
            _obs = obs["state"]
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs
    def select_action(self, obs, mode="visual"):
        _obs = self._obs_to_input(obs, mode)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()
    def sample_action(self, obs, mode="visual"):
        _obs = self._obs_to_input(obs, mode)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    # according to the update method in AAC
    # you put data to buffer, then you sample from buffer and update A, 
    def update_critic(self, obs, action, reward, next_actor_obs, next_critic_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_actor_obs)
            target_Q1, target_Q2 = self.critic_target(next_critic_obs, policy_action)
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

    def update_actor_and_alpha(self, actor_obs, critic_obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(actor_obs)
        actor_Q1, actor_Q2 = self.critic(critic_obs, pi)

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

    def default_update(self, replay_buffer, L, step):
        if self.use_aug:
            obs, action, reward, next_obs, not_done = replay_buffer.aug_sample()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
        visual = obs["visual"]
        state = obs["state"]
        next_visual = next_obs["visual"]
        next_state = next_obs["state"]
        self.update_critic(state, action, reward, next_visual, next_state, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(visual, state, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
    
    # use in BC distillation ? two buffer?