from utils import ReplayBuffer
import numpy as np
import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF
import random
import PIL.Image as Image

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
    obs, actions, mus, log_stds, rewards, next_obs, not_dones = buffer.behavior_costom_aug_sample(random_conv, random_crop, gaussian, add_random_color_patch)
    images = obs["visual"] # 128, 9, 100, 100

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

if __name__ == "__main__":
    aug_test()
    # sample_test()