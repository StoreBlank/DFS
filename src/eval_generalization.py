import os
import hydra
import torch
import metaworld
from env.custom_random_metaworld import ALL_V2_ENVIRONMENTS, ALL_TASKS
import random
import numpy as np
from datetime import datetime
import pandas as pd

import utils
from env.wrapper_metaworld import wrap
from video import VideoRecorder
from ipdb import set_trace

os.environ["MUJOCO_GL"] = "egl"


def evaluate(env_id, random_level, agent, video, num_episodes, video_name):
    # Initialize environments
    def make_env():
        env = ALL_V2_ENVIRONMENTS[env_id](random_level=random_level)
        task = random.choice(ALL_TASKS[f'level_{random_level}'][env_id])
        env.set_task(task)
        env = wrap(
            env,
            frame_stack=3,
            image_size=140,
        )
        return env

    episode_rewards = []
    num_success = 0
    for i in range(num_episodes):
        env = make_env()
        obs = env.reset()
        video.init(enabled=(i == 0))
        # video.init(enabled=True)
        done = False
        episode_reward = 0
        info = None
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save(f"{video_name}.mp4")
        episode_rewards.append(episode_reward)
        if info['success']:
            num_success += 1
        env.close()
    return np.mean(episode_rewards), num_success / num_episodes

@hydra.main(version_base=None, config_path="../configs", config_name="eval_generalization_config")
def main(args):
    algos = args.algos
    save_video = args.save_video
    eval_episodes = args.eval_episodes
    log_dir = args.log_dir
    env_config = args.env
    if not hasattr(env_config, 'camera_id'):
        env_config.camera_id = 0

    env_id = env_config.env_id
    utils.set_seed_everywhere(args.seed)

    work_dir = os.path.join(
        log_dir,
        env_id,
        'generalization_eval',
        str(datetime.now()),
    )
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if save_video else None, height=448, width=448, camera_id=env_config.camera_id)
    df = pd.DataFrame(columns=[1, 2, 3, 4], index=list(algos.keys()))
    
    for random_level in [1, 2, 3, 4]:
        print('random level:', random_level)
        for algo, path in algos.items():
            print(f"--- Loading {algo} weights from {path} ---")
            agent = torch.load(path)
            video_name = f"{algo}_{random_level}"
            # reward, _ = evaluate(env, task_id, agent, video, eval_episodes, video_name)
            # print(f"Reward: {reward}")
            # df.loc[algo, env_id] = reward
            _, success_rate = evaluate(env_id, random_level, agent, video, eval_episodes, video_name)
            print(f"Success rate: {success_rate}")
            df.loc[algo, random_level] = success_rate
    
    df.to_csv(os.path.join(work_dir, "eval.csv"))


if __name__ == "__main__":
    main()
