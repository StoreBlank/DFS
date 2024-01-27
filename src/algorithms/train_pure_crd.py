import torch
import os
import numpy as np
import utils
import time
import random
import metaworld
from env.custom_random_metaworld import ALL_V2_ENVIRONMENTS, ALL_TASKS
from env.wrapper_metaworld import wrap
from agents.bc_agent import PureCrdBC
from logger import Logger
from datetime import datetime
from video import VideoRecorder
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
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


def feature_show(agent, buffer, feat_num, save_path):
    feats_s, feats_t = agent.sample_feats(buffer, n=feat_num)
    print("Begin fitting")
    # tsne = TSNE(n_components=2, random_state=0)
    # feats_reduction = tsne.fit_transform(np.concatenate((feat_s, feat_t), axis=0))
    # plt.scatter(feats_reduction[:feat_num, 0], feats_reduction[:feat_num, 1], c='r')
    # plt.scatter(feats_reduction[feat_num:, 0], feats_reduction[feat_num:, 1], c='b')
    # plt.savefig(save_path)
    for i in range(len(feats_t)):
        tsne = TSNE(n_components=2, random_state=0)
        feats_reduction = tsne.fit_transform(np.concatenate((feats_s[i], feats_t[i]), axis=0))
        plt.scatter(feats_reduction[:feat_num, 0], feats_reduction[:feat_num, 1], c='r')
        plt.scatter(feats_reduction[feat_num:, 0], feats_reduction[feat_num:, 1], c='b')
        plt.savefig(save_path.replace(".png", f"_{i}.png"))


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
    feature_dir = utils.make_dir(os.path.join(work_dir, "feature"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(
        video_dir if algo_config.save_video else None, height=448, width=448, camera_id=env_config.camera_id
    )
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer.load(expert_config.buffer_path)
    # teacher
    teacher = torch.load(expert_config.model_path)
    teacher.eval()
    # student
    env_temp = initialize_env(env_id)
    cropped_visual_obs_shape = (
        env_temp.observation_space['visual'].shape[0],
        algo_config.image_crop_size,
        algo_config.image_crop_size,
    )
    agent = PureCrdBC(
        agent_obs_shape=cropped_visual_obs_shape,
        action_shape=env_temp.action_space.shape,
        agent_config=agent_config,
    )
    env_temp.close()
    agent.set_expert(teacher)

    # train features
    print("===============Training agent features===============")
    L = Logger(work_dir, use_wandb=algo_config.use_wandb)
    start_time = time.time()
    for step in range(algo_config.agent_feat_steps + 1):
        L.log("train/pure_crd_feat_duration", time.time() - start_time, step)
        start_time = time.time()
        L.dump(step)

        if step <= 100 and step % 10 == 0:
            print("Saving features")
            feature_show(agent, replay_buffer, algo_config.feat_num, os.path.join(feature_dir, f"agent_{step}.png"))

        agent.update(replay_buffer, L, step)
    
    # final features
    print("Saving features")
    feature_show(agent, replay_buffer, algo_config.feat_num, os.path.join(feature_dir, f"agent.png"))
    print("Agent features done")

    # save
    reloads = agent.release()
    torch.save(agent, os.path.join(model_dir, "pure_no_last.pt"))
    agent.reload_for_training(reloads)

    # agent last layer
    print("===============Training agent last layer===============")
    # freeze agent features
    agent.actor.freeze_feature_layers()
    start_time = time.time()
    for step in range(algo_config.agent_last_layer_steps + 1):
        L.log("train/pure_crd_last_duration", time.time() - start_time, step)
        start_time = time.time()
        L.dump(step)

        if step % algo_config.agent_eval_freq == 0:
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

        agent.update_last_layer(replay_buffer, L, step)

    # save
    agent.release()
    torch.save(agent, os.path.join(model_dir, "pure_crd.pt"))

    print("Done!")
