import torch
import os
import numpy as np
import gym
import utils
import time
from env.wrappers import make_env
from tqdm import tqdm

def collect_buffer(args):
    # parse config
    env_config = args.env
    agent_config = args.agent
    algo_config = args.algo

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
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

    # Create working directory
    work_dir = os.path.join(
        algo_config.log_dir,
        env_config.domain_name + "_" + env_config.task_name,
        args.algorithm,
    )
    print("Working directory:", work_dir)
    utils.make_dir(work_dir)

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        action_shape=env.action_space.shape,
        capacity=algo_config.rollout_steps,
        batch_size=algo_config.batch_size,
    )
    print(f"Loading expert weights from {agent_config.model_path} ...")
    agent=torch.load(agent_config.model_path)
    for param in agent.actor.parameters():
        param.requires_grad = False
    print("Expert loaded!")

    start_step, episode, episode_reward, done = 0, 0, 0, True
    for step in tqdm(range(start_step, algo_config.rollout_steps + 1), desc="Rollout Progress"):
        if done:
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        with utils.eval_mode(agent):
            action = agent.sample_action(obs)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    buffer_dir = utils.make_dir(work_dir)
    replay_buffer.save(os.path.join(buffer_dir, "replay_buffer.pkl"))

    print("Completed training for", work_dir)
