import os
import hydra
import torch
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import random
import numpy as np
from datetime import datetime
import pandas as pd

import utils
from env.wrapper_metaworld import wrap
from video import VideoRecorder
from ipdb import set_trace

os.environ["MUJOCO_GL"] = "egl"


def evaluate(env_id, agent, video, num_episodes, video_name, env_config):
    # Initialize environments
    mt10 = metaworld.MT10()
    if env_config.robust:
        def initialize_env(env_id):
            env = mt10.train_classes[env_id]()
            task = random.choice([task for task in mt10.train_tasks
                            if task.env_name == env_id])
            env.set_task(task)
            env = wrap(
                env,
                frame_stack=env_config.frame_stack,
                mode=env_config.mode,
                image_size=env_config.image_size,
            )
            return env
    else:
        def initialize_env(env_id):
            env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_id}-goal-observable']
            env = env_class()
            env = wrap(
                env,
                frame_stack=env_config.frame_stack,
                mode=env_config.mode,
                image_size=env_config.image_size,
            )
            return env

    episode_rewards = []
    num_success = 0
    for i in range(num_episodes):
        env = initialize_env(env_id)
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


@hydra.main(version_base=None, config_path="../configs", config_name="eval_state_policy_config")
def main(cfg):
    algos = cfg.algos
    save_video = cfg.save_video
    eval_episodes = cfg.eval_episodes
    log_dir = cfg.log_dir
    env_config = cfg.env
    if not hasattr(env_config, 'camera_id'):
        env_config.camera_id = 0

    env_ids = env_config.env_ids
    utils.set_seed_everywhere(cfg.seed)

    work_dir = os.path.join(
        log_dir,
        'multitask',
        "eval",
        str(datetime.now()),
    )
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if save_video else None, height=448, width=448, camera_id=env_config.camera_id)
    df = pd.DataFrame(columns=env_config.env_ids, index=['state_policy'])
    
    for env_id in env_ids:
        print(f"Env id: {env_id}")
        agent = torch.load(algos[env_id])
        video_name = f"{env_id}"
        _, success_rate = evaluate(env_id, agent, video, eval_episodes, video_name, env_config)
        print(f"Success rate: {success_rate}")
        df.loc['state_policy', env_id] = success_rate
    
    df.to_csv(os.path.join(work_dir, "eval.csv"))


if __name__ == "__main__":
    main()
