import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transform
import numpy as np
import os
import glob
import json
import random
import pickle
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
from ipdb import set_trace


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    with torch.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        "timestamp": str(datetime.now()),
        # "git": subprocess.check_output(["git", "describe", "--always"])
        # .strip()
        # .decode(),
        "args": OmegaConf.to_container(args, resolve=True),
    }
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def load_config(key=None):
    path = os.path.join("configs", "config.cfg")
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype="jpg", sort=True):
    fpath = os.path.join(dir_path, f"*.{filetype}")
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (
        w1 is not None and h1 is not None
    ), "must either specify both w1 and h1 or neither of them"
    assert isinstance(x, torch.Tensor) and x.is_cuda, "input must be CUDA tensor"

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped

def gaussian(x, mean=0., std=0.02):
    """Additive Gaussian noise"""
    return x + torch.randn_like(x) * std + mean

def random_conv(images):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    b, c, h, w = images.shape
    for i in range(b):
        for j in range(3):
            weights = torch.randn(3, 3, 3, 3).to(images.device)
            temp_image = images[i:i + 1][3*j:3*j+3].reshape(-1, 3, h, w) / 255.
            temp_image = F.pad(temp_image, pad=[1] * 4, mode='replicate')
            out = torch.sigmoid(F.conv2d(temp_image, weights)) * 255.
            total_out = out if i == 0 and j == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(b, c, h, w)

def add_random_color_patch(images, patch_size=24):
    batch_size, channels, height, width = images.size()
    for i in range(batch_size):
        for j in range(3): # three with different patch
            x = random.randint(0, width - patch_size)
            y = random.randint(0, height - patch_size)
            color = (random.random()*255, random.random()*255, random.random()*255) 
            image = images[i][3*j:3*j+3]
            image = TF.to_pil_image(image)
            patch = TF.pil_to_tensor(image.crop((x, y, x+patch_size, y+patch_size)))
            patch[:, :, :] = torch.tensor(color).view(3, 1, 1)
            image.paste(TF.to_pil_image(patch), (x, y))
            images[i][3*j:3*j+3] = TF.to_tensor(image)
    return images*255

def random_affine(images):
    batch_size, channels, height, width = images.size()
    for i in range(batch_size):
        for j in range(3):
            image = images[i][3*j:3*j+3]
            image = TF.to_pil_image(image)
            fill_color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))
            output = transform.RandomAffine(degrees=(0,20), translate=(0.1, 0.2), shear=(0,20), fill=fill_color)(image)
            images[i][3*j:3*j+3] = TF.to_tensor(output)
    return images*255

def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(
        x.shape
    ), "window_shape must be a tuple with same number of dimensions as x"

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3),
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, action_shape, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.mus = np.empty((capacity, *action_shape), dtype=np.float32)
        self.log_stds = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.not_dones = np.empty((capacity,), dtype=np.float32)

        self.idx = 0
        self.full = False

    def save(self, file_path):
        print(f"Saving replay buffer to {file_path} ...")
        with open(file_path, 'wb') as fi:
            pickle.dump(self, fi)
        print("Replay buffer saved!")

    @staticmethod
    def load(file_path):
        print(f"Loading replay buffer from {file_path} ...")
        with open(file_path, 'rb') as fi:
            obj = pickle.load(fi)
        print("Replay buffer loaded!")
        return obj
    
    def add(self, obs, action, reward, next_obs, done):
        obses = (obs.copy(), next_obs.copy())
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = obses
        np.copyto(self.actions[self.idx], action)
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = not done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_behavior(self, obs, action, mu, log_std, reward, next_obs, done):
        obses = (obs.copy(), next_obs.copy())
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = obses
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.mus[self.idx], mu)
        np.copyto(self.log_stds[self.idx], log_std)
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = not done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(0, self.capacity if self.full else self.idx, size=n)

    def _encode_obses(self, idxs):
        obses = {"state": [], "visual": []}
        next_obses = {"state": [], "visual": []}
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses["state"].append(np.array(obs["state"], copy=False))
            obses["visual"].append(np.array(obs["visual"], copy=False))
            next_obses["state"].append(np.array(next_obs["state"], copy=False))
            next_obses["visual"].append(np.array(next_obs["visual"], copy=False))
        obses["state"] = np.array(obses["state"])
        obses["visual"] = np.array(obses["visual"])
        next_obses["state"] = np.array(next_obses["state"])
        next_obses["visual"] = np.array(next_obses["visual"])
        return obses, next_obses

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = {
            "state": torch.as_tensor(obs["state"]).cuda().float(),
            "visual": torch.as_tensor(obs["visual"]).cuda().float(),
        }
        next_obs = {
            "state": torch.as_tensor(next_obs["state"]).cuda().float(),
            "visual": torch.as_tensor(next_obs["visual"]).cuda().float(),
        }
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        mus = torch.as_tensor(self.mus[idxs]).cuda()
        log_stds = torch.as_tensor(self.log_stds[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs

    # def sample_svea(self, n=None, pad=4):
    #     obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
    #     # obs = augmentations.random_shift(obs, pad)
    #     augs = transforms.Compose([
    #         transforms.RandomPerspective(),
    #         transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    #     ])
    #     obs = augs(obs)

    #     return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None):
        obs, actions, _, _, rewards, next_obs, not_dones, _ = self.__sample__(n=n)

        return obs, actions, rewards, next_obs, not_dones

    def aug_sample(self, n=None):
        obs, actions, _, _, rewards, next_obs, not_dones, _ = self.__sample__(n=n)
        obs["visual"] = random_crop(obs["visual"])
        next_obs["visual"] = random_crop(next_obs["visual"])

        return obs, actions, rewards, next_obs, not_dones
    
    def costom_aug_sample(self, func, n=None):
        obs, actions, _, _, rewards, next_obs, not_dones, _ = self.__sample__(n=n)
        obs["visual"] = func(obs["visual"])
        next_obs["visual"] = func(next_obs["visual"])

        return obs, actions, rewards, next_obs, not_dones
    
    def behavior_sample(self, n=None, return_idxs=False):
        obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs = self.__sample__(n=n)

        if return_idxs:
            return obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs
        return obs, actions, mus, log_stds, rewards, next_obs, not_dones
    
    def behavior_aug_sample(self, n=None, return_idxs=False):
        obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs = self.__sample__(n=n)
        obs["visual"] = random_crop(obs["visual"])
        next_obs["visual"] = random_crop(next_obs["visual"])

        if return_idxs:
            return obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs
        return obs, actions, mus, log_stds, rewards, next_obs, not_dones

    def behavior_costom_aug_sample(self, *funcs, n=None, return_idxs=False):
        obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs = self.__sample__(n=n)
        func = random.choice(funcs)
        obs["visual"] = func(obs["visual"])
        next_obs["visual"] = func(next_obs["visual"])

        if return_idxs:
            return obs, actions, mus, log_stds, rewards, next_obs, not_dones, idxs
        return obs, actions, mus, log_stds, rewards, next_obs, not_dones


def collect_buffer(agent, env, rollout_steps, batch_size, work_dir):
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = ReplayBuffer(
        action_shape=env.action_space.shape,
        capacity=rollout_steps,
        batch_size=batch_size,
    )
    
    start_step, episode, episode_reward, done = 0, 0, 0, True
    for step in tqdm(range(start_step, rollout_steps + 1), desc="Rollout Progress"):
        if done:
            obs = env.reset()
            done = False
            print(f"Episode {episode} reward: {episode_reward}")
            episode_reward = 0
            episode += 1

        with torch.no_grad():
            mu, pi, log_std = agent.exhibit_behavior(obs)

        # Take step
        next_obs, reward, done, _ = env.step(pi)
        replay_buffer.add_behavior(obs, pi, mu, log_std, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

    if work_dir is not None:
        buffer_dir = make_dir(work_dir)
        replay_buffer.save(os.path.join(buffer_dir, f"{rollout_steps}.pkl"))

    print("Completed rollout")
    return replay_buffer


class ContrastBuffer(ReplayBuffer):
    """Buffer for normal transitions and corresponding features"""
    def __init__(self, action_shape, capacity, batch_size, feature_dim, s_layers, t_layers, K, T=0.07, momentum=0.5):
        super().__init__(action_shape, capacity, batch_size)
        self.K = K
        self.T = T
        self.Z_v1 = -np.ones(s_layers)
        self.Z_v2 = -np.ones(t_layers)
        self.momentum = momentum

        stdv = 1. / np.sqrt(feature_dim / 3)
        self.memory_v1 = torch.rand(s_layers, capacity, feature_dim).mul_(2 * stdv).add_(-stdv).cuda()
        self.memory_v2 = torch.rand(t_layers, capacity, feature_dim).mul_(2 * stdv).add_(-stdv).cuda()

    # def load_buffer(self, buffer: ReplayBuffer):
    #     self._obses = buffer._obses
    #     self.actions = buffer.actions
    #     self.mus = buffer.mus
    #     self.log_stds = buffer.log_stds
    #     self.rewards = buffer.rewards
    #     self.not_dones = buffer.not_dones
    #     self.idx = buffer.idx
    #     self.full = buffer.full

    @staticmethod
    def load(file_path, opt):
        print(f"Loading replay buffer from {file_path} ...")
        with open(file_path, 'rb') as fi:
            obj = pickle.load(fi)
        print("Replay buffer loaded!")
        contrastive_buffer = ContrastBuffer(
            action_shape=obj.actions.shape[1:],
            capacity=obj.capacity,
            batch_size=obj.batch_size,
            feature_dim=opt.feature_dim,
            s_layers=opt.s_layers,
            t_layers=opt.t_layers,
            K=opt.K,
            T=opt.T,
            momentum=opt.momentum,
        )
        contrastive_buffer._obses = obj._obses
        contrastive_buffer.actions = obj.actions
        contrastive_buffer.mus = obj.mus
        contrastive_buffer.log_stds = obj.log_stds
        contrastive_buffer.rewards = obj.rewards
        contrastive_buffer.not_dones = obj.not_dones
        contrastive_buffer.idx = obj.idx
        contrastive_buffer.full = obj.full

        print("Contrastive wrapped!")
        return contrastive_buffer

    def contrast(self, f_s, f_t, s_layer, t_layer, idx, contrast_idx=None):
        if contrast_idx is None:
            contrast_idx = self._get_idxs((len(idx), self.K))
        idx = np.concatenate([idx[:, None], contrast_idx], 1)
        
        # teacher side
        weight_s = self.memory_v1[s_layer][idx]
        out_t = torch.bmm(weight_s, f_t.unsqueeze(2)).squeeze(2)
        out_t = torch.exp(out_t / self.T)
        # student side
        weight_t = self.memory_v2[t_layer][idx]
        out_s = torch.bmm(weight_t, f_s.unsqueeze(2)).squeeze(2)
        out_s = torch.exp(out_s / self.T)

        # set Z if haven't been set yet
        # if self.Z_v1 < 0:
        #     self.Z_v1 = out_s.mean().detach().cpu().numpy() * self.capacity
        # if self.Z_v2 < 0:
        #     self.Z_v2 = out_t.mean().detach().cpu().numpy() * self.capacity
        if self.Z_v1[s_layer] < 0:
            self.Z_v1[s_layer] = out_s.mean().detach().cpu().numpy() * self.capacity
        if self.Z_v2[t_layer] < 0:
            self.Z_v2[t_layer] = out_t.mean().detach().cpu().numpy() * self.capacity

        out_s = (out_s / self.Z_v1[s_layer]).contiguous()
        out_t = (out_t / self.Z_v2[t_layer]).contiguous()

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

        return out_s, out_t
