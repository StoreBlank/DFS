import torch
import os
import numpy as np
import utils
import time
import random
import metaworld
from env.custom_random_metaworld import ALL_V2_ENVIRONMENTS, ALL_TASKS
from env.wrapper_metaworld import wrap
from agents.bc_agent import BC
from logger import Logger
from datetime import datetime
from video import VideoRecorder
from ipdb import set_trace


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
    """
    Args:
        num_episodes: how many episodes for eval
        L: logger
        step: which step in the whole training process
    return:
        mean reward of episodes
    """
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
    expert_config = args.expert
    algo_config = args.algo
    if not algo_config.crop:
        algo_config.image_crop_size = env_config.image_size
    if not hasattr(env_config, 'camera_id'):
        env_config.camera_id = 0

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Initialize environments for evaluation
    env_id = env_config.env_id
    if env_config.category == 'metaworld':
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
            from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
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
    else:
        if env_config.robust:
            random_level = env_config.random_level
            def initialize_env(env_id):
                env_cls = ALL_V2_ENVIRONMENTS[env_id]
                env = env_cls(random_level=random_level)
                task = random.choice(ALL_TASKS[f'level_{random_level}']['door-close-v2'])
                env.set_task(task)
                env = wrap(
                    env,
                    frame_stack=env_config.frame_stack,
                    mode=env_config.mode,
                    image_size=env_config.image_size,
                    done_on_success=False,
                )
                return env
        else:
            from env.custom_random_metaworld import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
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

    # Create working directory
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
        video_dir if algo_config.save_video else None, height=448, width=448, camera_id=env_config.camera_id
    )
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer.load(expert_config.buffer_path)
    # student
    env_temp = initialize_env(env_id)
    cropped_visual_obs_shape = (
        env_temp.observation_space['visual'].shape[0],
        algo_config.image_crop_size,
        algo_config.image_crop_size,
    )
    agent = BC(
        agent_obs_shape=cropped_visual_obs_shape,
        action_shape=env_temp.action_space.shape,
        agent_config=agent_config,
    )
    env_temp.close()
    
    # train
    print("Training student")
    start_step = 0
    L = Logger(work_dir, use_wandb=algo_config.use_wandb)
    start_time = time.time()
    for step in range(start_step, algo_config.train_steps + 1):
        if step > start_step:
            L.log("train/duration", time.time() - start_time, step)
            start_time = time.time()
            L.dump(step)

        # Evaluate agent periodically
        if step % algo_config.eval_freq == 0:
            print("Evaluating:", work_dir)
            env = initialize_env(env_id)
            evaluate(
                env,
                agent,
                video,
                algo_config.eval_episodes,
                L,
                step,
            )
            env.close()
            L.dump(step)

        # Save agent periodically
        if step > start_step and step % algo_config.save_freq == 0:
            torch.save(agent, os.path.join(model_dir, f"{step}.pt"))

        # Run training update
        agent.update(replay_buffer, L, step)

    print("Completed training for", work_dir)
