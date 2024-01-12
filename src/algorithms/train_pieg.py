import torch
import os
import numpy as np
import utils
import time
from env.wrappers import make_env
from agents.pieg_agent import PIEGAgent
from logger import Logger
from datetime import datetime
from video import VideoRecorder

def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
    episode_rewards = []
    _test_env = "_test_env" if test_env else ""
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

        if L is not None:
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        video.save(f"{step}{_test_env}.mp4")
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)

def train(args):
    # parse config
    env_config = args.env
    agent_config = args.agent
    algo_config = args.algo
    algo_config.image_crop_size = 84 if algo_config.crop else 100

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Initialize environments
    if env_config.category == 'dmc':
        env = make_env(
            category=env_config.category,
            domain_name=env_config.domain_name,
            task_name=env_config.task_name,
            seed=algo_config.seed,
            episode_length=env_config.episode_length,
            action_repeat=env_config.action_repeat,
            image_size=env_config.image_size,
            frame_stack=env_config.frame_stack,
            # mode="train",
            mode="distracting_cs",
            intensity=env_config.distracting_cs_intensity,
        )
        test_env = (
            None
            if env_config.eval_mode is None
            else make_env(
                category=env_config.category,
                domain_name=env_config.domain_name,
                task_name=env_config.task_name,
                seed=algo_config.seed + 42,
                episode_length=env_config.episode_length,
                action_repeat=env_config.action_repeat,
                image_size=env_config.image_size,
                frame_stack=env_config.frame_stack,
                mode=env_config.eval_mode,
                intensity=env_config.distracting_cs_intensity,
            )
        )
    elif env_config.category == 'maniskill':
        env = make_env(
            category=env_config.category,
            env_id=env_config.env_id,
            frame_stack=env_config.frame_stack,
            control_mode=env_config.control_mode,
            renderer_kwargs=env_config.renderer_kwargs,
        )
        test_env = None

    # Create working directory
    if env_config.category == 'dmc':
        env_config.env_id = env_config.domain_name + "_" + env_config.task_name
    work_dir = os.path.join(
        algo_config.log_dir,
        env_config.env_id,
        args.algorithm,
        str(algo_config.seed),
        str(datetime.now()),
    )
    print("Working directory:", work_dir)
    assert not os.path.exists(
        os.path.join(work_dir, "train.log")
    ), "specified working directory already exists"
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(
        video_dir if algo_config.save_video else None, height=448, width=448
    )
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        action_shape=env.action_space.shape,
        capacity=algo_config.train_steps,
        batch_size=algo_config.batch_size,
    )
    cropped_visual_obs_shape = (
        env.observation_space['visual'].shape[0],
        algo_config.image_crop_size,
        algo_config.image_crop_size,
    )

    agent = PIEGAgent(
        obs_shape=cropped_visual_obs_shape,
        action_shape=env.action_space.shape,
        args=agent_config,
    ) # TODO: change init in PIEGAgent
    
