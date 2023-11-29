import os
import hydra
import torch
import numpy as np
from datetime import datetime
import pandas as pd

import utils
from env.wrappers import make_env
from video import VideoRecorder
from ipdb import set_trace

os.environ["MUJOCO_GL"] = "egl"


def evaluate(env, agent, video, num_episodes, video_name):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save(f"{video_name}.mp4")
        episode_rewards.append(episode_reward)
        print("saved!")
    return np.mean(episode_rewards)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg):
    algos = cfg.algos
    intensities = cfg.intensities
    save_video = cfg.save_video
    eval_episodes = cfg.eval_episodes
    env_config = cfg.env

    work_dir = f"./logs/{env_config.domain_name}_{env_config.task_name}/eval/{str(datetime.now())}/"
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if save_video else None, height=448, width=448)
    df = pd.DataFrame(columns=intensities, index=list(algos.keys()))

    for intensity in intensities:
        print(f"Init env with intensity {intensity}")
        env = make_env(
            domain_name=env_config.domain_name,
            task_name=env_config.task_name,
            seed=42,
            episode_length=env_config.episode_length,
            action_repeat=env_config.action_repeat,
            image_size=env_config.image_size,
            frame_stack=env_config.frame_stack,
            mode="distracting_cs",
            intensity=intensity,
        )
        for algo, path in algos.items():
            print(f"--- Loading {algo} weights from {path} ---")
            agent = torch.load(path)
            video_name = f"{algo}_{intensity}"
            reward = evaluate(env, agent, video, eval_episodes, video_name)
            print(f"Got reward {reward}")
            df[algo, intensity] = reward

    df.to_csv(os.path.join(work_dir, "result.csv"))


if __name__ == "__main__":
    main()
