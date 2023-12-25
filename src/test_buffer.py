from utils import ReplayBuffer
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transform
import random
import PIL.Image as Image
from tqdm import trange

from utils import random_crop

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
            output = transform.RandomAffine(degrees=(0,30), translate=(0.1, 0.2), shear=(0,30), fill=fill_color)(image)
            images[i][3*j:3*j+3] = TF.to_tensor(output)
    return images*255


def visualize(images, start=0):
    # 128, 9, 100, 100 where 9 is three continous frame
    image_array = np.array(images[0].cpu())[0:3]
    image_array = np.transpose(image_array, (1, 2, 0))[:, :, ::-1]
    image = Image.fromarray(np.uint8(image_array))
    image.save(f"test{start}.png")

    image_array = np.array(images[0].cpu())[3:6]
    image_array = np.transpose(image_array, (1, 2, 0))[:, :, ::-1]
    image = Image.fromarray(np.uint8(image_array))
    image.save(f"test{start+1}.png")

    image_array = np.array(images[0].cpu())[6:]
    image_array = np.transpose(image_array, (1, 2, 0))[:, :, ::-1]
    image = Image.fromarray(np.uint8(image_array))
    image.save(f"test{start+2}.png")

def aug_test():
    buffer=ReplayBuffer.load("./buffers/50000_v2.pkl")
    obs, actions, mus, log_stds, rewards, next_obs, not_dones = buffer.behavior_costom_aug_sample(random_conv, random_crop, add_random_color_patch)
    images = obs["visual"] # 128, 9, 100, 100

    images = random_affine(images)

    # k=random.random()
    # if k<0.5:
    #     images=random_conv(images)
    # else:
    #     images = gaussian(images)
    #     images = add_random_color_patch(images)
    #     images = random_crop(images)


    visualize(images, start=3)

def sample_test():
    buffer=ReplayBuffer.load("./logs/finger_spin/visualbc_buffer_collector/42/2023-12-13 20:24:51.953302/5000.pkl")
    obs, actions, rewards, next_obs, not_dones = buffer.sample()
    print(obs["visual"].shape)
    print(next_obs["visual"].shape)

def encoder_self_mse_test():
    buffer=ReplayBuffer.load("./logs/finger_spin/visualbc_buffer_collector/42/2023-12-13 20:24:51.953302/5000.pkl")
    crdbc=torch.load("./logs/finger_spin/crd_bc/42/2023-12-14 09:46:36.810582/model/200000.pt")
    bc=torch.load("./logs/finger_spin/vanilla_bc/42/2023-12-14 13:40:35.229162/model/200000.pt")
    
    crdbc_encoder = crdbc.actor.encoder
    bc_encoder = bc.actor.encoder

    crd_loss = 0
    bc_loss = 0

    for i in trange(30):
        obs, _, _, _, _ = buffer.sample()
        obs_visual = obs["visual"]
        auged_obs_visual = obs["visual"].clone()

        if i <15:
            auged_obs_visual = random_conv(auged_obs_visual)
        else:
            auged_obs_visual = add_random_color_patch(auged_obs_visual)

        with torch.no_grad():
            crd_feat1 = crdbc_encoder(obs_visual, detach=True)
            crd_feat2 = crdbc_encoder(auged_obs_visual, detach=True)
            bc_feat1 = bc_encoder(obs_visual, detach=True)
            bc_feat2 = bc_encoder(auged_obs_visual, detach=True)
        
        crdbc_encoder_loss = F.mse_loss(crd_feat1, crd_feat2)
        bc_encoder_loss = F.mse_loss(bc_feat1, bc_feat2)
        crd_loss+=crdbc_encoder_loss
        bc_loss+=bc_encoder_loss

    print(crd_loss/30)
    print(bc_loss/30)

if __name__ == "__main__":
    aug_test()
    # sample_test()
    # encoder_self_mse_test()