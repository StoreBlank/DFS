import torch
import os
import numpy as np
import utils
import time
import random
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from env.wrapper_metaworld import wrap
from agents.sac_agent import StateSAC, VisualSAC, NoisyStateSAC
from logger import Logger
from datetime import datetime
from video import VideoRecorder
from ipdb import set_trace


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
    if not algo_config.crop:
        algo_config.image_crop_size = env_config.image_size
    if not hasattr(env_config, 'camera_id'):
        env_config.camera_id = 0

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Initialize environments
    if env_config.category == 'metaworld':
        test_env = None
        mt1 = metaworld.MT1(env_config.env_id)
        env_class = mt1.train_classes[env_config.env_id]
        def make_env():
            env = env_class()
            task = random.choice(mt1.train_tasks)
            env.set_task(task)
            env = wrap(
                env,
                frame_stack=env_config.frame_stack,
                mode=env_config.mode,
                image_size=env_config.image_size,
            )
            return env
        # env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_config.env_id}-goal-observable']
        # env = env_class()
        # env = wrap(
        #     env,
        #     frame_stack=env_config.frame_stack,
        #     mode=env_config.mode,
        #     image_size=env_config.image_size,
        # )
        # test_env = None

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
    env = make_env()
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
    # print("Observations:", env.observation_space.shape)
    # print("Cropped observations:", cropped_obs_shape)
    if algo_config.mode == "state":
        agent = StateSAC(
            obs_shape=env.observation_space["state"].shape,
            action_shape=env.action_space.shape,
            args=agent_config,
        )
    elif algo_config.mode == "noisy_state":
        agent = NoisyStateSAC(
            obs_shape=env.observation_space["state"].shape,
            action_shape=env.action_space.shape,
            args=agent_config,
        )
    else:
        agent = VisualSAC(
            obs_shape=cropped_visual_obs_shape,
            action_shape=env.action_space.shape,
            args=agent_config,
        )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir, use_wandb=algo_config.use_wandb)
    start_time = time.time()
    for step in range(start_step, algo_config.train_steps + 1):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % algo_config.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(
                    env,
                    agent,
                    video,
                    algo_config.eval_episodes,
                    L,
                    step,
                )
                if test_env is not None:
                    evaluate(
                        test_env,
                        agent,
                        video,
                        algo_config.eval_episodes,
                        L,
                        step,
                        test_env=True,
                    )
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % algo_config.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f"{step}.pt"))

            L.log("train/episode_reward", episode_reward, step)

            # obs = env.reset()
            env.close()
            env = make_env()
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

        # Sample action for data collection
        if step < algo_config.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= algo_config.init_steps:
            num_updates = (
                algo_config.init_steps if step == algo_config.init_steps else 1
            )
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

        # if step == start_step:
        # debug for augmentation
        # print('obs: ', obs.frame(0))
        # save first obs
        # os.makedirs(os.path.join(work_dir, 'debug'), exist_ok=True)
        # utils.save_image(obs.frame(0),
        #                  os.path.join(work_dir, 'debug', 'obs.png'))
        # tensor = torch.tensor(obs.frame(0)).clone().cuda().unsqueeze(0)
        # start_time = time.time()
        # aug_tensor = augmentations.random_overlay(
        #     augmentations.random_crop(tensor))
        # print('aug time: ', time.time() - start_time)
        # np_array = aug_tensor.squeeze(0).cpu().numpy()
        # utils.save_image(
        #     np_array, os.path.join(work_dir, 'debug', 'obs_aug.png'))

        episode_step += 1
    
    print("Completed training for", work_dir)
    env.close()
