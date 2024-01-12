import numpy as np
import math
import os
import glob
import json
import random
import pickle
import abc
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transform
from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
from time import time

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

    def __sample__(self, n=None, use_loader=False, loader_idx=None):
        if use_loader:
            assert loader_idx != None
            idxs = [loader_idx]
        else:
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

    def sample(self, n=None, use_loader=False, loader_idx=None):
        # if you want to put the buffer into a Dataloader, set use_loader=True, and set loader_idx as use idx in __getitem__
        obs, actions, _, _, rewards, next_obs, not_dones, _ = self.__sample__(n=n, use_loader=use_loader, loader_idx=loader_idx)

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

"""
Code Borrowed From StableKeypoints
"""
"""
1. attention control register
"""
class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, dict, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, dict, is_cross: bool, place_in_unet: str):
        
        dict = self.forward(dict, is_cross, place_in_unet)
        
        return dict['attn']

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "attn": [],
        }

    def forward(self, dict, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 32**2:  # avoid memory overhead
        self.step_store["attn"].append(dict['attn']) 
        
        return dict

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()

def register_attention_control(model, controller, feature_upsample_res=256):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None

            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            # sim = torch.matmul(q, k.permute(0, 2, 1)) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim = sim.masked_fill(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = torch.nn.Softmax(dim=-1)(sim)
            attn = attn.clone()
            
            out = torch.matmul(attn, v)

            if (
                is_cross
                and sequence_length <= 32**2
                and len(controller.step_store["attn"]) < 4
            ):
                x_reshaped = x.reshape(
                    batch_size,
                    int(sequence_length**0.5),
                    int(sequence_length**0.5),
                    dim,
                ).permute(0, 3, 1, 2)
                # upsample to feature_upsample_res**2
                x_reshaped = (
                    F.interpolate(
                        x_reshaped,
                        size=(feature_upsample_res, feature_upsample_res),
                        mode="bicubic",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size, -1, dim)
                )

                q = self.to_q(x_reshaped)
                q = self.reshape_heads_to_batch_dim(q)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
                attn = torch.nn.Softmax(dim=-1)(sim)
                attn = attn.clone()

                attn = controller({"attn": attn}, is_cross, place_in_unet)

            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")

    controller.num_att_layers = cross_att_count
    
    # create assertion with message
    assert cross_att_count != 0, "No cross attention layers found in the model. Please check to make sure you're using diffusers==0.8.0."


"""
2. model loader with register
"""
def load_ldm(device, type="CompVis/stable-diffusion-v1-4", feature_upsample_res=256):
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    MY_TOKEN = ""
    NUM_DDIM_STEPS = 50
    scheduler.set_timesteps(NUM_DDIM_STEPS)


    ldm = StableDiffusionPipeline.from_pretrained(
        type, use_auth_token=MY_TOKEN, scheduler=scheduler, local_files_only=True
    ).to(device)
    
    if device != "cpu":
        ldm.unet = nn.DataParallel(ldm.unet)
        ldm.vae = nn.DataParallel(ldm.vae)
        
        controllers = {}
        for device_id in ldm.unet.device_ids:
            device = torch.device("cuda", device_id)
            controller = AttentionStore()
            controllers[device] = controller
    else:
        controllers = {}
        _device = torch.device("cpu")
        controller = AttentionStore()
        controllers[_device] = controller

        # patched_devices = set()

    def hook_fn(module, input):
        _device = input[0].device
        # if device not in patched_devices:
        register_attention_control(module, controllers[_device], feature_upsample_res=feature_upsample_res)
        # patched_devices.add(device)

    if device != "cpu":
        ldm.unet.module.register_forward_pre_hook(hook_fn)
    else:
        ldm.unet.register_forward_pre_hook(hook_fn)
    
    num_gpus = torch.cuda.device_count()

    for param in ldm.vae.parameters():
        param.requires_grad = False
    for param in ldm.text_encoder.parameters():
        param.requires_grad = False
    for param in ldm.unet.parameters():
        param.requires_grad = False

    return ldm, controllers, num_gpus

"""
3. help class for keypoint finding
"""
class RandomAffineWithInverse:
    def __init__(
        self,
        degrees=0,
        scale=(1.0, 1.0),
        translate=(0.0, 0.0),
    ):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate

        # Initialize self.last_params to 0s
        self.last_params = {
            "theta": torch.eye(2, 3).unsqueeze(0),
        }

    def create_affine_matrix(self, angle, scale, translations_percent):
        angle_rad = math.radians(angle)

        # Create affine matrix
        theta = torch.tensor(
            [
                [math.cos(angle_rad), math.sin(angle_rad), translations_percent[0]],
                [-math.sin(angle_rad), math.cos(angle_rad), translations_percent[1]],
            ],
            dtype=torch.float,
        )

        theta[:, :2] = theta[:, :2] * scale
        theta = theta.unsqueeze(0)  # Add batch dimension
        return theta

    def __call__(self, img_tensor, theta=None):

        if theta is None:
            theta = []
            for i in range(img_tensor.shape[0]):
                # Calculate random parameters
                angle = torch.rand(1).item() * (2 * self.degrees) - self.degrees
                scale_factor = torch.rand(1).item() * (self.scale[1] - self.scale[0]) + self.scale[0]
                translations_percent = (
                    torch.rand(1).item() * (2 * self.translate[0]) - self.translate[0],
                    torch.rand(1).item() * (2 * self.translate[1]) - self.translate[1],
                    # 1.0,
                    # 1.0,
                )

                # Create the affine matrix
                theta.append(self.create_affine_matrix(
                    angle, scale_factor, translations_percent
                ))
            theta = torch.cat(theta, dim=0).to(img_tensor.device)

        # Store them for inverse transformation
        self.last_params = {
            "theta": theta,
        }

        # Apply transformation
        grid = F.affine_grid(theta, img_tensor.size(), align_corners=False).to(
            img_tensor.device
        )
        transformed_img = F.grid_sample(img_tensor, grid, align_corners=False)

        return transformed_img

    def inverse(self, img_tensor):

        # Retrieve stored parameters
        theta = self.last_params["theta"]

        # Augment the affine matrix to make it 3x3
        theta_augmented = torch.cat(
            [theta, torch.Tensor([[0, 0, 1]]).expand(theta.shape[0], -1, -1)], dim=1
        )

        # Compute the inverse of the affine matrix
        theta_inv_augmented = torch.inverse(theta_augmented)
        theta_inv = theta_inv_augmented[:, :2, :]  # Take the 2x3 part back

        # Apply inverse transformation
        grid_inv = F.affine_grid(theta_inv, img_tensor.size(), align_corners=False).to(
            img_tensor.device
        )
        untransformed_img = F.grid_sample(img_tensor, grid_inv, align_corners=False)

        return untransformed_img

"""
4. class we use for get mask
---
usage:
    edit param in config file for agent training
    you need first get embedding.pt, indices.pt
    you can get this by running StableKeypoint/unsupervised/main.py (just the 1,2 stage is OK)
"""
class Keypoint_HardMask(object):
    def __init__(
        self, 
        ldm_path,
        embedding_path,
        indices_path,
        device,
        # model usage and output
        num_points = 10,
        from_where = ["down_cross", "mid_cross", "up_cross"],
        layers = [0,1,2,3],
        noise_level = -1,
        max_loc_strategy="argmax",
        # augmentation for keypoint
        mask_scale = 13, 
        augment_degrees=15.0,
        augment_scale=(0.8, 1.0),
        augment_translate=(0.25, 0.25),
        augmentation_iterations=20,
        # image
        feature_upsample_res = 128, # upsampled resolution for latent features grabbed from the attn operation
        image_size = 512, # for vae input
    ):
        self.ldm, self.controllers, self.num_gpus = load_ldm(device, ldm_path, feature_upsample_res)
        self.embeddings = torch.load(embedding_path).to(device).detach()
        self.indices = torch.load(indices_path).to(device).detach()
        self.device = device

        self.transform = transform.Compose([
            transform.Resize(image_size),
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.num_points = num_points
        self.from_where = from_where
        self.layers = layers
        self.noise_level = noise_level
        self.max_loc_strategy = max_loc_strategy

        self.mask_scale = mask_scale
        self.augment_degrees = augment_degrees
        self.augment_scale = augment_scale
        self.augment_translate = augment_translate
        self.augmentation_iterations = augmentation_iterations

        self.image_size = image_size

    @torch.no_grad()
    def get_mask(self, img):
        """
        img: [3, w, h] numpy array from obs["visual"], but we need first change it to [w,h,3] to align
        """
        assert img.shape[1] == img.shape[2]
        start_time = time()

        img = img.transpose(1,2,0) # convert to image like input
        origin_image_size = img.shape[1]
        img = transform.ToPILImage()(img)
        img = self.transform(img) # to [3, 512, 512] tensor on CPU

        map = self._run_image_with_context_augmented(img)

        if self.max_loc_strategy == "argmax":   
            points = self._find_max_pixel(map.view(self.num_points, 512, 512)) / 512.0 # [num_points, 2] TODO: or we put this into MLP in RL
        else:
            points = self._pixel_from_weighted_avg(map.view(self.num_points, 512, 512)) / 512.0
        points = points.reshape(self.num_points, 2) # note the points here is in [0,1], size: [10, 2] on cuda

        # set_trace()
        
        points = origin_image_size*points
        mask_size = self.mask_scale * origin_image_size//100

        mask = torch.zeros(origin_image_size, origin_image_size) # same to original size, not 512x512
        for i in range(self.num_points):
            y, x = points[i]
            x = int(x)
            y = int(y)

            start_x = max(0, x - mask_size // 2)
            end_x = min(origin_image_size, x + mask_size // 2)
            start_y = max(0, y - mask_size // 2)
            end_y = min(origin_image_size, y + mask_size // 2)

            mask[start_y:end_y, start_x:end_x] = 1

        mask.unsqueeze(0) # [1, 512, 512]

        end_time = time()
        print(f"mask got! time consuming: {end_time - start_time} s")

        return mask.to(torch.int8)

    def _find_max_pixel(self, map):
        """
        finds the pixel of the map with the highest value
        map shape [batch_size, h, w]
        
        output shape [batch_size, 2]
        """
        num_points, h, w = map.shape

        map_reshaped = map.view(num_points, -1)
        max_indices = torch.argmax(map_reshaped, dim=-1)
        max_indices = max_indices.view(num_points, 1)
        max_indices = torch.cat([max_indices // w, max_indices % w], dim=-1) # [num_points, 2] with axis
        
        max_indices = max_indices + 0.5 # offset by a half a pixel to get the center of the pixel

        return max_indices
    
    def _pixel_from_weighted_avg(self, heatmaps, distance=5):
        """
        finds the pixel of the map with the highest value
        map shape [batch_size, h, w]
        """
        # Get the shape of the heatmaps
        batch_size, m, n = heatmaps.shape
        # If distance is provided, zero out elements outside the distance from the max pixel
        if distance != -1:
            # Find max pixel using your existing function or similar logic
            max_pixel_indices = self._find_max_pixel(heatmaps)
            x_max, y_max = max_pixel_indices[:, 0].long(), max_pixel_indices[:, 1].long()
            # Create a meshgrid
            x = torch.arange(0, m).float().view(1, m, 1).to(heatmaps.device).repeat(batch_size, 1, 1)
            y = torch.arange(0, n).float().view(1, 1, n).to(heatmaps.device).repeat(batch_size, 1, 1)
            # Calculate the distance to the max_pixel
            distance_to_max = torch.sqrt((x - x_max.view(batch_size, 1, 1)) ** 2 + 
                                        (y - y_max.view(batch_size, 1, 1)) ** 2)
            # Zero out elements beyond the distance
            heatmaps[distance_to_max > distance] = 0.0
        # Compute the total value of the heatmaps
        total_value = torch.sum(heatmaps, dim=[1, 2], keepdim=True)
        # Normalize the heatmaps
        normalized_heatmaps = heatmaps / (
            total_value + 1e-6
        )  # Adding a small constant to avoid division by zero
        # Create meshgrid to represent the coordinates
        x = torch.arange(0, m).float().view(1, m, 1).to(heatmaps.device)
        y = torch.arange(0, n).float().view(1, 1, n).to(heatmaps.device)
        # Compute the weighted sum for x and y
        x_sum = torch.sum(x * normalized_heatmaps, dim=[1, 2])
        y_sum = torch.sum(y * normalized_heatmaps, dim=[1, 2])

        return torch.stack([x_sum, y_sum], dim=-1) + 0.5

    @torch.no_grad()
    def _run_image_with_context_augmented(self, image):
        # if image is a torch.tensor, convert to numpy
        if type(image) == torch.Tensor:
            image = image.permute(1, 2, 0).detach().cpu().numpy()

        num_samples = torch.zeros(len(self.indices), 512, 512).to(self.device)
        sum_samples = torch.zeros(len(self.indices), 512, 512).to(self.device)

        invertible_transform = RandomAffineWithInverse(
            degrees=self.augment_degrees,
            scale=self.augment_scale,
            translate=self.augment_translate,
        )

        for i in range(self.augmentation_iterations//self.num_gpus):
            
            augmented_img = (
                invertible_transform(torch.tensor(image)[None].repeat(self.num_gpus, 1, 1, 1).permute(0, 3, 1, 2))
                .permute(0, 2, 3, 1)
                .numpy()
            )
            
            attn_maps = self._run_and_find_attn(augmented_img,upsample_res=512)
            
            attn_maps = torch.stack([map.to(self.device) for map in attn_maps])
            
            _num_samples = invertible_transform.inverse(torch.ones_like(attn_maps))
            _sum_samples = invertible_transform.inverse(attn_maps)

            num_samples += _num_samples.sum(dim=0)
            sum_samples += _sum_samples.sum(dim=0)

        # visualize sum_samples/num_samples
        attention_maps = sum_samples / num_samples

        # replace all nans with 0s
        attention_maps[attention_maps != attention_maps] = 0

        return attention_maps
    
    def _run_and_find_attn(self, image, upsample_res=32):
        _, _ = self._find_pred_noise(image)
        
        attention_maps=[]
        
        for controller in self.controllers:

            _attention_maps = self._collect_maps(
                self.controllers[controller],
                upsample_res=upsample_res,
            )
            
            attention_maps.append(_attention_maps)

            self.controllers[controller].reset()

        return attention_maps

    def _find_pred_noise(self, image):
        # if image is a torch.tensor, convert to numpy
        if type(image) == torch.Tensor:
            image = image.permute(0, 2, 3, 1).detach().cpu().numpy()

        with torch.no_grad():
            latent = self._image2latent(image)
            
        noise = torch.randn_like(latent)

        noisy_image = self.ldm.scheduler.add_noise( latent, noise, self.ldm.scheduler.timesteps[self.noise_level])

        pred_noise = self.ldm.unet(noisy_image, 
                            self.ldm.scheduler.timesteps[self.noise_level].repeat(noisy_image.shape[0]), 
                            self.embeddings.repeat(noisy_image.shape[0], 1, 1))["sample"]
        
        return noise, pred_noise

    def _image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                # print the max and min values of the image
                image = torch.from_numpy(image).float() * 2 - 1
                image = image.permute(0, 3, 1, 2).to(self.device)
                if self.device != "cpu":
                    latents = self.ldm.vae.module.encode(image)["latent_dist"].mean
                else:
                    latents = self.ldm.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    def _collect_maps(self, controller, upsample_res=512):
        """
        returns the bilinearly upsampled attention map of size upsample_res x upsample_res for the first word in the prompt
        """

        attention_maps = controller.step_store['attn']
        attention_maps_list = []
        layer_overall = -1

        # import ipdb; ipdb.set_trace()
        for layer in range(len(attention_maps)):
            layer_overall += 1
            if layer_overall not in self.layers:
                continue
            data = attention_maps[layer]
            data = data.reshape(
                data.shape[0], int(data.shape[1] ** 0.5), int(data.shape[1] ** 0.5), data.shape[2]
            )
            
            # import ipdb; ipdb.set_trace()
            if self.indices is not None:
                data = data[:, :, :, self.indices]

            data = data.permute(0, 3, 1, 2)

            if upsample_res != -1 and data.shape[1] ** 0.5 != upsample_res:
                # bilinearly upsample the image to attn_sizexattn_size
                data = F.interpolate(
                    data,
                    size=(upsample_res, upsample_res),
                    mode="bilinear",
                    align_corners=False,
                )

            attention_maps_list.append(data)

        attention_maps_list = torch.stack(attention_maps_list, dim=0).mean(dim=(0, 1))

        controller.reset()

        return attention_maps_list