import argparse
import os
import PIL.Image as Image
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import ReplayBuffer

def save_image(images, path):
    image_array = np.array(images[0].cpu())[0:3]
    image_array = np.transpose(image_array, (1, 2, 0))[:, :, ::-1]
    image = Image.fromarray(np.uint8(image_array))
    image.save(path)

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder", type=str, help="model folder")
    parser.add_argument("-b", "--buffer_path", type=str, help="buffer path")
    args = parser.parse_args()

    model_paths = os.listdir(args.model_folder)

    buffer = ReplayBuffer.load(args.buffer_path)
    obs, _, _, _, _  = buffer.sample()
    images = obs["visual"] # 128, 9, 100, 100

    save_image(images, "./output/origin.png")

    for model_path in model_paths:
        model_path = os.path.join(args.model_folder, model_path)
        get_feature_map(model_path, images)