import torch
import os
import numpy as np
import utils
import time
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from env.wrapper_metaworld import wrap
from env.wrappers import make_env
from agents.bc_agent import MultitaskCrdBC
from logger import Logger
from datetime import datetime
from video import VideoRecorder
from ipdb import set_trace


def evaluate(env, task_id, task_name, agent, video, num_episodes, L, step, test_env=False):
    episode_rewards = []
    _test_env = "_test_env" if test_env else ""
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs, task_id)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        if L is not None:
            L.log(f"eval/{task_name}_episode_reward{_test_env}", episode_reward, step)
        video.save(f"{task_name}_{step}{_test_env}.mp4")
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

    # Initialize environments
    num_task = env_config.num_task
    env_ids = env_config.env_ids
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
        'multitask',
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
    replay_buffers = []
    for env_id in env_ids:
        replay_buffer = utils.ReplayBuffer.load(expert_config.buffer_paths[env_id])
        replay_buffers.append(replay_buffer)
    # teacher
    teachers = []
    for env_id in env_ids:
        teacher = torch.load(expert_config.model_paths[env_id])
        teacher.eval()
        teachers.append(teacher)
    # student
    # temp env for spaces
    env_temp = initialize_env(env_ids[0])
    cropped_visual_obs_shape = (
        env_temp.observation_space['visual'].shape[0],
        algo_config.image_crop_size,
        algo_config.image_crop_size,
    )
    agent = MultitaskCrdBC(
        agent_obs_shape=cropped_visual_obs_shape,
        action_shape=env_temp.action_space.shape,
        agent_config=agent_config,
    )
    env_temp.close()
    agent.set_experts(teachers)
    # agent.prefill_memory(replay_buffers)

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
            for task_id, task_name in enumerate(env_ids):
                env = initialize_env(task_name)
                evaluate(
                    env,
                    task_id,
                    task_name,
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
        for task_id in range(num_task):
            agent.update(replay_buffers[task_id], task_id, L, step)

    print("Completed training for", work_dir)
