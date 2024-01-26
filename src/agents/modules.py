import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace


LOG_STD_MAX = 2
LOG_STD_MIN = -10
eps = 1e-7


def _get_out_shape_cuda(in_shape, layers):
    x = torch.randn(*in_shape).cuda().unsqueeze(0)
    return layers(x).squeeze(0).shape


def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1)
    return mu, pi, log_pi


# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     """Truncated normal distribution, see https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf"""
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#     with torch.no_grad():
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#         return tensor


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class CenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        # assert size in {84, 100}, f"unexpected size: {size}"
        self.size = size

    def forward(self, x):
        assert x.ndim == 4, "input must be a 4D tensor"
        B, C, H, W = x.shape
        assert H == W, "input must be a square tensor"
        assert H >= self.size, "input size must be larger than target size"
        assert (H - self.size) % 2 == 0, "crop must be even"
        p = (H - self.size) // 2
        if x.size(2) == self.size and x.size(3) == self.size:
            return x
        # assert x.size(3) == 100, f"unexpected size: {x.size(3)}"
        # if self.size == 84:
        #     p = 8
        return x[:, :, p:-p, p:-p]


class NormalizeImg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 255.0


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RLProjection(nn.Module):
    def __init__(self, in_shape, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_shape[0], out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


# class SODAMLP(nn.Module):
# 	def __init__(self, projection_dim, hidden_dim, out_dim):
# 		super().__init__()
# 		self.out_dim = out_dim
# 		self.mlp = nn.Sequential(
# 			nn.Linear(projection_dim, hidden_dim),
# 			nn.BatchNorm1d(hidden_dim),
# 			nn.ReLU(),
# 			nn.Linear(hidden_dim, out_dim)
# 		)
# 		self.apply(weight_init)

# 	def forward(self, x):
# 		return self.mlp(x)


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.layers = [
            CenterCrop(size=obs_shape[1]),
            NormalizeImg(),
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
        ]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(weight_init)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        return self.projection(x)


class VisualActor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.encoder = encoder
        self.layer_1 = nn.Linear(encoder.out_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.mu_layer = nn.Linear(hidden_dim, action_shape[0])
        self.log_std_layer = nn.Linear(hidden_dim, action_shape[0])
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)
        self.mu_layer.apply(weight_init)
        self.log_std_layer.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True, detach=False, feat=False, pre_act=False, encoder_task=False):
        x = self.encoder(x, detach)
        f1_pre = self.layer_1(x)
        f1 = self.act(f1_pre)
        f2_pre = self.layer_2(f1)
        f2 = self.act(f2_pre)
        f3_pre = self.layer_3(f2)
        f3 = self.act(f3_pre)
        mu = self.mu_layer(f3)
        log_std = self.log_std_layer(f3)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        if encoder_task:
            return x

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if pre_act:
            return mu, pi, log_pi, log_std, [f1_pre, f1, f2_pre, f2, f3_pre, f3]
        elif feat:
            return mu, pi, log_pi, log_std, [f1, f2, f3]
        else:
            return mu, pi, log_pi, log_std

    def feature_layers(self):
        return [self.encoder, self.layer_1, self.layer_2, self.layer_3, self.act]

    def last_layer(self):
        return [self.mu_layer, self.log_std_layer]

    def freeze_feature_layers(self):
        for layer in self.feature_layers():
            for param in layer.parameters():
                param.requires_grad = False


class MultitaskVisualActor(nn.Module):
    def __init__(self, encoder, num_task, action_shape, hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.num_task = num_task
        self.encoder = encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(num_task, 64),
            nn.Tanh()
        )
        self.layer_1 = nn.Linear(encoder.out_dim + 64, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.mu_layer = nn.Linear(hidden_dim, action_shape[0])
        self.log_std_layer = nn.Linear(hidden_dim, action_shape[0])
        self.task_encoder.apply(weight_init)
        self.layer_1.apply(weight_init)
        self.layer_2.apply(weight_init)
        self.layer_3.apply(weight_init)
        self.mu_layer.apply(weight_init)
        self.log_std_layer.apply(weight_init)

    def forward(self, x, task_ids, compute_pi=True, compute_log_pi=True, detach=False, feat=False, pre_act=False):
        """
        x: a batch of observations
        task_ids: corresponding task ids, (B,)
        """
        x = self.encoder(x, detach)
        task_vec = self.task_encoder(F.one_hot(task_ids, self.num_task).float())
        x = torch.cat([x, task_vec], dim=1)
        f1_pre = self.layer_1(x)
        f1 = self.act(f1_pre)
        f2_pre = self.layer_2(f1)
        f2 = self.act(f2_pre)
        f3_pre = self.layer_3(f2)
        f3 = self.act(f3_pre)
        mu = self.mu_layer(f3)
        log_std = self.log_std_layer(f3)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if pre_act:
            return mu, pi, log_pi, log_std, [f1_pre, f1, f2_pre, f2, f3_pre, f3]
        elif feat:
            return mu, pi, log_pi, log_std, [f1, f2, f3]
        else:
            return mu, pi, log_pi, log_std
        
    def feature_layers(self):
        return [self.encoder, self.task_encoder, self.layer_1, self.layer_2, self.layer_3, self.act]
    
    def last_layer(self):
        return [self.mu_layer, self.log_std_layer]
    
    def freeze_feature_layers(self):
        for layer in self.feature_layers():
            for param in layer.parameters():
                param.requires_grad = False


class StateActor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.layer_1 = nn.Linear(obs_shape[0], hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.mu_layer = nn.Linear(hidden_dim, action_shape[0])
        self.log_std_layer = nn.Linear(hidden_dim, action_shape[0])
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, feat=False, pre_act=False):
        f1_pre = self.layer_1(obs)
        f1 = self.act(f1_pre)
        f2_pre = self.layer_2(f1)
        f2 = self.act(f2_pre)
        f3_pre = self.layer_3(f2)
        f3 = self.act(f3_pre)
        mu = self.mu_layer(f3)
        log_std = self.log_std_layer(f3)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if pre_act:
            return mu, pi, log_pi, log_std, [f1_pre, f1, f2_pre, f2, f3_pre, f3]
        elif feat:
            return mu, pi, log_pi, log_std, [f1, f2, f3]
        else:
            return mu, pi, log_pi, log_std


class NoisyStateActor(nn.Module):
    """
    Augment the state observation with Gaussian noise.
    """
    def __init__(self, obs_shape, action_shape, hidden_dim, auged_obs_dim, aug, activation=nn.ReLU):
        super().__init__()
        self.layer_1 = nn.Linear(auged_obs_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation()
        self.mu_layer = nn.Linear(hidden_dim, action_shape[0])
        self.log_std_layer = nn.Linear(hidden_dim, action_shape[0])
        self.auged_obs_dim = auged_obs_dim
        self.aug = aug
        if aug == "linear":
            self.aug_layer = nn.Linear(auged_obs_dim, auged_obs_dim)
            # freeze the aug_layer
            for param in self.aug_layer.parameters():
                param.requires_grad = False
        self.apply(weight_init)

    def _aug_obs(self, obs):
        bsz = obs.shape[0]
        noise = torch.randn(bsz, self.auged_obs_dim - obs.shape[1]).to(obs.device)
        auged_obs = torch.cat([obs, noise], dim=1)
        if self.aug == "shuffle":
            auged_obs = auged_obs[:, torch.randperm(auged_obs.shape[1])]
            return auged_obs
        elif self.aug == "linear":
            auged_obs = self.aug_layer(auged_obs)
            return auged_obs
        else:
            return auged_obs
        
    def forward(self, obs, compute_pi=True, compute_log_pi=True, feat=False, pre_act=False):
        auged_obs = self._aug_obs(obs)
        f1_pre = self.layer_1(auged_obs)
        f1 = self.act(f1_pre)
        f2_pre = self.layer_2(f1)
        f2 = self.act(f2_pre)
        mu = self.mu_layer(f2)
        log_std = self.log_std_layer(f2)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if pre_act:
            return mu, pi, log_pi, log_std, [f1_pre, f1, f2_pre, f2]
        elif feat:
            return mu, pi, log_pi, log_std, [f1, f2]
        else:
            return mu, pi, log_pi, log_std


class MLPQFunction(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, activation):
        super().__init__()
        self.q = mlp(
            [obs_shape[0] + action_shape[0]] + [hidden_dim, hidden_dim, hidden_dim] + [1],
            activation,
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        q = self.q(torch.cat([obs, action], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class StateCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.q1 = MLPQFunction(obs_shape, action_shape, hidden_dim, activation)
        self.q2 = MLPQFunction(obs_shape, action_shape, hidden_dim, activation)

    def forward(self, obs, action):
        q1 = self.q1(obs, action)
        q2 = self.q2(obs, action)
        return q1, q2


# class QFunction(nn.Module):
# 	def __init__(self, obs_shape, action_shape, hidden_dim):
# 		super().__init__()
# 		self.trunk = nn.Sequential(
# 			nn.Linear(obs_shape + action_shape, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, 1)
# 		)
# 		self.apply(weight_init)

# 	def forward(self, obs, action):
# 		assert obs.size(0) == action.size(0)
# 		return self.trunk(torch.cat([obs, action], dim=1))


class VisualCritic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, activation=nn.ReLU):
        super().__init__()
        self.encoder = encoder
        self.q1 = MLPQFunction((encoder.out_dim,), action_shape, hidden_dim, activation)
        self.q2 = MLPQFunction((encoder.out_dim,), action_shape, hidden_dim, activation)

    def forward(self, x, action, detach=False):
        x = self.encoder(x, detach)
        return self.q1(x, action), self.q2(x, action)


# class CURLHead(nn.Module):
# 	def __init__(self, encoder):
# 		super().__init__()
# 		self.encoder = encoder
# 		self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

# 	def compute_logits(self, z_a, z_pos):
# 		"""
# 		Uses logits trick for CURL:
# 		- compute (B,B) matrix z_a (W z_pos.T)
# 		- positives are all diagonal elements
# 		- negatives are all other elements
# 		- to compute loss use multiclass cross entropy with identity matrix for labels
# 		"""
# 		Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
# 		logits = torch.matmul(z_a, Wz)  # (B,B)
# 		logits = logits - torch.max(logits, 1)[0][:, None]
# 		return logits


# class InverseDynamics(nn.Module):
# 	def __init__(self, encoder, action_shape, hidden_dim):
# 		super().__init__()
# 		self.encoder = encoder
# 		self.mlp = nn.Sequential(
# 			nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
# 			nn.Linear(hidden_dim, action_shape[0])
# 		)
# 		self.apply(weight_init)

# 	def forward(self, x, x_next):
# 		h = self.encoder(x)
# 		h_next = self.encoder(x_next)
# 		joint_h = torch.cat([h, h_next], dim=1)
# 		return self.mlp(joint_h)


# class SODAPredictor(nn.Module):
# 	def __init__(self, encoder, hidden_dim):
# 		super().__init__()
# 		self.encoder = encoder
# 		self.mlp = SODAMLP(
# 			encoder.out_dim, hidden_dim, encoder.out_dim
# 		)
# 		self.apply(weight_init)

# 	def forward(self, x):
# 		return self.mlp(self.encoder(x))


class ContrastMemory(nn.Module):
    """
    Memory buffer stored in model
    """
    def __init__(
            self,
            capacity,
            feature_dim,
            s_layers,
            t_layers,
            K,
            T=0.07,
            momentum=0.5
        ):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.K = K
        self.momentum = momentum

        stdv = 1. / np.sqrt(feature_dim / 3)
        self.register_buffer('memory_v1', torch.rand(s_layers, capacity, feature_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(t_layers, capacity, feature_dim).mul_(2 * stdv).add_(-stdv))

    def _get_and_store(
            self,
            f_s,
            f_t,
            s_layer,
            t_layer,
            idx,
            contrast_idx=None
        ):
        """
        Get negative sample and store the positive sample
        Return:
            weight_s: (B, K + 1, f_dim)
            weight_t: (B, K + 1, f_dim)
        """
        if contrast_idx is None:
            contrast_idx = np.random.randint(self.capacity, size=(len(idx), self.K))
        idx = np.concatenate([idx[:, None], contrast_idx], 1)

        # update memory
        with torch.no_grad():
            for i, ind in enumerate(idx[:, 0]):
                s_pos = self.memory_v1[s_layer][ind] * self.momentum + f_s[i] * (1. - self.momentum)
                s_norm = torch.linalg.norm(s_pos)
                updated_s = s_pos / s_norm
                self.memory_v1[s_layer][ind] = updated_s

                t_pos = self.memory_v2[t_layer][ind] * self.momentum + f_t[i] * (1. - self.momentum)
                t_norm = torch.linalg.norm(t_pos)
                updated_t = t_pos / t_norm
                self.memory_v2[t_layer][ind] = updated_t

        weight_s = self.memory_v1[s_layer][idx]
        weight_t = self.memory_v2[t_layer][idx]

        return weight_s, weight_t

    def forward(
            self,
            f_s,
            f_t,
            s_layer,
            t_layer,
            idx,
            contrast_idx=None
        ):
        return self._get_and_store(f_s, f_t, s_layer, t_layer, idx, contrast_idx)


def contrast_loss(x, residual):
    # loss for positive pair
    P_pos = x[:, 0]
    log_D1 = torch.log(P_pos / (P_pos + residual + eps))

    # loss for negative pairs
    P_neg = x[:, 1:]
    log_D0 = torch.log(residual / (P_neg + residual + eps)).sum(1)

    assert log_D1.dim() == log_D0.dim() == 1
    loss = - (log_D1 + log_D0).mean()

    return loss


class CRDLoss(nn.Module):
    """
    CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dims: the dimension of student's features
        opt.t_dims: the dimension of teacher's features
        opt.feat_dim: the dimension of the projection space
        opt.crd_weight: the weight of cross layer contrastive loss
        opt.nce_k: number of negatives paired with each positive
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
        opt.T: temperature
        opt.momentum: momentum for updating memory
    """
    def __init__(self, opt):
        super().__init__()
        # self.embed_s = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(opt.s_dim, opt.feat_dim),
        # )
        self.embeds_s = []
        for s_dim in opt.s_dims:
            self.embeds_s.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(s_dim, opt.feat_dim),
            ))
        self.embeds_s = nn.ModuleList(self.embeds_s).cuda()
        # self.embed_t = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(opt.t_dim, opt.feat_dim),
        # )
        self.embeds_t = []
        for t_dim in opt.t_dims:
            self.embeds_t.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(t_dim, opt.feat_dim),
            ))
        self.embeds_t = nn.ModuleList(self.embeds_t).cuda()
        self.crd_weight = opt.crd_weight
        self.residual = opt.nce_k / opt.n_data
        self.T = opt.T
        self.memory_in_gpu = opt.contrastive_memory_in_gpu
        self.memory = ContrastMemory(
            opt.n_data,
            opt.feat_dim,
            len(opt.s_dims),
            len(opt.t_dims),
            opt.nce_k,
            opt.T,
            opt.momentum
        )
        if self.memory_in_gpu:
            self.memory.cuda()
        self.Z_v1 = -np.ones(len(opt.s_dims))
        self.Z_v2 = -np.ones(len(opt.t_dims))
        self.apply(weight_init)

    def contrast(self, f_s, f_t, s_layer, t_layer, idx, contrast_idx=None):
        """
        Single layer contrast loss
        """
        f_s = self.embeds_s[s_layer](f_s)
        f_s = F.normalize(f_s, dim=1)
        f_t = self.embeds_t[t_layer](f_t)
        f_t = F.normalize(f_t, dim=1)

        if self.memory_in_gpu:
            weight_s, weight_t = self.memory(f_s, f_t, s_layer, t_layer, idx, contrast_idx)
        else:
            weight_s, weight_t = self.memory(f_s.cpu(), f_t.cpu(), s_layer, t_layer, idx, contrast_idx)
            weight_s = weight_s.cuda()
            weight_t = weight_t.cuda()

        # teacher side
        out_t = torch.bmm(weight_s, f_t.unsqueeze(2)).squeeze(2)
        out_t = torch.exp(out_t / self.T)
        # student side
        out_s = torch.bmm(weight_t, f_s.unsqueeze(2)).squeeze(2)
        out_s = torch.exp(out_s / self.T)
        
        if self.Z_v1[s_layer] < 0:
            self.Z_v1[s_layer] = out_s.mean().detach().cpu().numpy() * self.memory.capacity
        if self.Z_v2[t_layer] < 0:
            self.Z_v2[t_layer] = out_t.mean().detach().cpu().numpy() * self.memory.capacity

        out_s = (out_s / self.Z_v1[s_layer]).contiguous()
        out_t = (out_t / self.Z_v2[t_layer]).contiguous()

        s_loss = contrast_loss(out_s, self.residual)
        t_loss = contrast_loss(out_t, self.residual)

        return s_loss + t_loss

    def forward(self, fs_s, fs_t, idx, contrast_idx=None):
        """
        Args:
            fs_s: the features of student network, a list, each element size [batch_size, s_dim]
            fs_t: the features of teacher network, a list, each element size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        # f_s = self.embed_s(f_s)
        # f_t = self.embed_t(f_t)
        # f_s = F.normalize(f_s, dim=1)
        # f_t = F.normalize(f_t, dim=1)
        # out_s, out_t = buffer.contrast(f_s, f_t, idx, contrast_idx) # take out contrast pair
        loss = 0
        # for logging
        losses = np.zeros((len(fs_s), len(fs_t)))
        for i in range(len(fs_s)):
            for j in range(len(fs_t)):
                if self.crd_weight[i][j] != 0:
                    layer_loss = self.contrast(fs_s[i], fs_t[j], i, j, idx, contrast_idx) # take out contrast pair
                    loss += self.crd_weight[i][j] * layer_loss
                    losses[i][j] = layer_loss.detach().cpu().numpy()
        # s_loss = contrast_loss(out_s, self.residual)
        # t_loss = contrast_loss(out_t, self.residual)
        # loss = s_loss + t_loss
        return loss, losses


class HalfProjContrastMemory(nn.Module):
    def __init__(
            self,
            capacity,
            feature_dim,
            t_layers,
            K,
            T=50
        ):
        super().__init__()
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.K = K

        self.register_buffer('memory', torch.rand(t_layers, capacity, feature_dim))

    def prefill(self, teacher, buffer):
        print("Prefilling contrastive memory ...")
        with torch.no_grad():
            max_batch = buffer.capacity // buffer.batch_size
            for i in range(max_batch - 1):
                idxs = list(range(i * buffer.batch_size, min((i + 1) * buffer.batch_size, buffer.capacity)))
                obses, _  = buffer._encode_obses(idxs)
                obs = {
                    "state": torch.as_tensor(obses["state"]).cuda().float(),
                    "visual": torch.as_tensor(obses["visual"]).cuda().float(),
                }
                _, _, _, _, feats_t = teacher.actor(obs['state'], False, False, True, True)
                feats_t = torch.stack(feats_t)
                self.memory[:, idxs, :] = feats_t
        print("Contrastive memory pre-filled!")

    def _get(
            self,
            t_layer,
            idx,
            contrast_idx=None
        ):
        """
        only get negative sample
        Return:
            weight: (B, K + 1, f_dim)
        """
        if contrast_idx is None:
            contrast_idx = np.random.randint(self.capacity, size=(len(idx), self.K))
        idx = np.concatenate([idx[:, None], contrast_idx], 1)

        weight = self.memory[t_layer][idx]

        return weight
    
    def forward(
            self,
            t_layer,
            idx,
            contrast_idx=None
        ):
        return self._get(t_layer, idx, contrast_idx)


def l1_dist(f_s, weight_t):
    """
    l1 distance
    Args:
        f_s: (B, f_dim)
        weight_t: (B, K + 1, f_dim)
    Return:
        dist: (B, K + 1)
    """
    return torch.cdist(f_s.unsqueeze(1), weight_t, p=1).squeeze(1)

def l2_dist(f_s, weight_t):
    """
    l2 distance
    Args:
        f_s: (B, f_dim)
        weight_t: (B, K + 1, f_dim)
    Return:
        dist: (B, K + 1)
    """
    return torch.cdist(f_s.unsqueeze(1), weight_t, p=2).squeeze(1)


class HalfProjCRDLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.embeds_s = []
        for s_dim in opt.s_dims:
            self.embeds_s.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(s_dim, opt.feat_dim),
            ))
        self.embeds_s = nn.ModuleList(self.embeds_s).cuda()
        self.crd_weight = opt.crd_weight
        self.residual = opt.nce_k / opt.n_data
        self.T = opt.T
        self.memory_in_gpu = opt.contrastive_memory_in_gpu
        self.memory = HalfProjContrastMemory(
            opt.n_data,
            opt.feat_dim,
            len(opt.t_dims),
            opt.nce_k,
            opt.T
        )
        if self.memory_in_gpu:
            self.memory.cuda()
        self.Z = -np.ones(len(opt.t_dims))
        self.apply(weight_init)

    def contrast(self, f_s, f_t, s_layer, t_layer, idx, contrast_idx=None):
        """
        Single layer contrast loss
        """
        f_s = self.embeds_s[s_layer](f_s)

        if self.memory_in_gpu:
            weight_t = self.memory(t_layer, idx, contrast_idx)
        else:
            weight_t = self.memory(t_layer, idx, contrast_idx)
            weight_t = weight_t.cuda()

        out = l2_dist(f_s, weight_t)
        out = torch.exp(-out / self.T)
        
        if self.Z[t_layer] < 0:
            self.Z[t_layer] = out.mean().detach().cpu().numpy() * self.memory.capacity

        out = (out / self.Z[t_layer]).contiguous()

        loss = contrast_loss(out, self.residual)

        return loss
    
    def forward(self, fs_s, fs_t, idx, contrast_idx=None):
        """
        Args:
            fs_s: the features of student network, a list, each element size [batch_size, s_dim]
            fs_t: the features of teacher network, a list, each element size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        # f_s = self.embed_s(f_s)
        # f_t = self.embed_t(f_t)
        # f_s = F.normalize(f_s, dim=1)
        # f_t = F.normalize(f_t, dim=1)
        # out_s, out_t = buffer.contrast(f_s, f_t, idx, contrast_idx) # take out contrast pair
        loss = 0
        # for logging
        losses = np.zeros((len(fs_s), len(fs_t)))
        for i in range(len(fs_s)):
            for j in range(len(fs_t)):
                if self.crd_weight[i][j] != 0:
                    layer_loss = self.contrast(fs_s[i], fs_t[j], i, j, idx, contrast_idx) # take out contrast pair
                    loss += self.crd_weight[i][j] * layer_loss
                    losses[i][j] = layer_loss.detach().cpu().numpy()
        # s_loss = contrast_loss(out_s, self.residual)
        # t_loss = contrast_loss(out_t, self.residual)
        # loss = s_loss + t_loss
        return loss, losses


# Calculate Gram matrix
def gram(x):
    bs, L = x.size()
    x_T = x.transpose(0, 1)
    G = x @ x_T / (bs * L)
    return G

# normalize 2d tensor such that each element has unit norm
def normalize_2d(x):
    mean = x.mean()
    std = (x - mean).pow(2).mean().sqrt()
    return (x - mean) / std


class EdgeSimLoss(nn.Module):
    """
    Edge similarity loss function
    """
    def __init__(self, opt):
        super().__init__()
        self.es_weight = opt.es_weight
        self.es_size = opt.es_size
        self.mean_s = -np.ones(len(opt.s_dims))
        self.std_s = -np.ones(len(opt.s_dims))
        self.mean_t = -np.ones(len(opt.t_dims))
        self.std_t = -np.ones(len(opt.t_dims))
        self.es_normalize = opt.es_normalize

    def single_layer_loss(self, f_s, f_t, s_layer, t_layer):
        # choose a random subset of size *size* of the batch
        idx = np.random.choice(f_s.shape[0], self.es_size, replace=False)
        f_s = f_s[idx]
        f_t = f_t[idx]

        # calculate gram matrix
        G_s = gram(f_s)
        G_t = gram(f_t)

        if self.mean_s[s_layer] < 0:
            self.mean_s[s_layer] = G_s.mean().detach().cpu().numpy()
        if self.std_s[s_layer] < 0:
            self.std_s[s_layer] = (G_s - G_s.mean()).pow(2).mean().sqrt().detach().cpu().numpy()
        if self.mean_t[t_layer] < 0:
            self.mean_t[t_layer] = G_t.mean().detach().cpu().numpy()
        if self.std_t[t_layer] < 0:
            self.std_t[t_layer] = (G_t - G_t.mean()).pow(2).mean().sqrt().detach().cpu().numpy()

        if self.es_normalize:
            G_s = normalize_2d(G_s)
            G_t = normalize_2d(G_t)
        else:
            G_s = (G_s - self.mean_s[s_layer]) / self.std_s[s_layer]
            G_t = (G_t - self.mean_t[t_layer]) / self.std_t[t_layer]

        # calculate loss
        loss = F.mse_loss(G_s, G_t)

        return loss

    def forward(self, fs_s, fs_t):
        loss = 0
        for i in range(len(fs_s)):
            for j in range(len(fs_t)):
                if self.es_weight[i][j] != 0:
                    layer_loss = self.single_layer_loss(fs_s[i], fs_t[j], i, j)
                    loss += self.es_weight[i][j] * layer_loss
        return loss
