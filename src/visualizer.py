import argparse
import os

import numpy as np
import PIL.Image as Image
import torch
import torchvision
import torch.utils.data
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import ReplayBuffer

def save_image(images, path):
    image_array = np.array(images[0].cpu())[0:3]
    image_array = np.transpose(image_array, (1, 2, 0))[:, :, ::-1]
    image = Image.fromarray(np.uint8(image_array))
    image.save(path)

def save_batch_image(images, num, path):
    imgs = images[0:num][:,0:3]/255.  #*255 or /255.
    img = torchvision.utils.make_grid(imgs, nrow=4)
    torchvision.utils.save_image(img, path)

def save_continue_images(images, path):
    _,_,w,h = images.shape
    imgs = (images[0]/255.).reshape(3,3,w,h)
    img = torchvision.utils.make_grid(imgs, nrow=3)
    torchvision.utils.save_image(img, path)

def get_feature_map(model_path, images):
    model = torch.load(model_path)
    layers = model.actor.encoder.shared_cnn.layers
    folder = model_path.split("_")[-2].split("/")[-1]
    save_folder = f"./output/{folder}"

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    feature_map_list = []

    for i in tqdm(range(len(layers))):
        if i%2 == 0:
            continue
        else:
            encoder = layers[0:i+1]
            with torch.no_grad():
                feature_maps = encoder(images).cpu() # 128x32x21x21

            b, c, w, h = feature_maps.shape
            flatten_features = feature_maps.view(b,c,-1).permute(0,2,1) # 128x441x32
            flatten_features = flatten_features.reshape(-1, c)
            pca = PCA(n_components=3)
            feature_maps = pca.fit_transform(flatten_features)
            feature_maps = feature_maps.reshape(b, w, h, 3)

            path = os.path.join(save_folder, f"{i//2+1}.png")
            feature_map = np.array(feature_maps[0])
            feature_map_list.append(feature_map)
            image = Image.fromarray(np.uint8(feature_map))
            image.save(path)

    return feature_map_list

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, buffer, image_size):
        super().__init__()
        self.buffer = buffer

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        obs, _, _, _, _  = self.buffer.sample(use_loader=True, loader_idx=idx)
        img = obs["visual"][0][0:3]
        img = transforms.ToPILImage()(img)
        sample = {'img': self.transform(img)}

        return sample

    def __len__(self):
        return self.buffer.capacity

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder", type=str, help="model folder")
    parser.add_argument("-b", "--buffer_path", type=str, help="buffer path")
    args = parser.parse_args()

    model_paths = os.listdir(args.model_folder)

    buffer = ReplayBuffer.load(args.buffer_path)
    obs, _, _, _, _  = buffer.sample(128)
    images = obs["visual"] # 128, 9, 100, 100
    print(images.device) # cuda:0

    # PCA visualizer
    # save_image(images, "./output/origin.png")
    # for model_path in model_paths:
    #     model_path = os.path.join(args.model_folder, model_path)
    #     get_feature_map(model_path, images)

    # batch visualizer
    # save_batch_image(images, 32, "./output/batch2.png")

    # # continue frame visualizer
    # save_continue_images(images, "./output/continue_frames.png")

    # # dataloader
    # dataset = TrainSet(buffer, 512)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    # dataloader_iter = iter(dataloader)
    # for i in range(1):
    #     batch = next(dataloader_iter) # batch["img"] 128x3x512x512
    #     imgs = batch["img"]
    #     # save_batch_image(imgs, 32, "./output/dataloader.png")