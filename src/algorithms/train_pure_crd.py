import torch
import os
import numpy as np
import gym
import utils
import time
from env.wrappers import make_env
from agents.bc_agent import FeatBaselineBC, PureCrdBC
from logger import Logger
from datetime import datetime
from video import VideoRecorder
from ipdb import set_trace
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


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
    feat_s, feat_t = agent.sample_feats(buffer, n=feat_num)
    print("Begin fitting")
    tsne = TSNE(n_components=2, random_state=0)
    feats_reduction = tsne.fit_transform(np.concatenate((feat_s, feat_t), axis=0))
    plt.scatter(feats_reduction[:feat_num, 0], feats_reduction[:feat_num, 1], c='r')
    plt.scatter(feats_reduction[feat_num:, 0], feats_reduction[feat_num:, 1], c='b')
    plt.savefig(save_path)


def train(args):
    # parse config
    env_config = args.env
    agent_config = args.agent
    expert_config = args.expert
    algo_config = args.algo
    baseline_config = args.baseline
    algo_config.image_crop_size = 84 if algo_config.crop else 100

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Initialize environments
    gym.logger.set_level(40)
    # env = make_env(
    #     domain_name=env_config.domain_name,
    #     task_name=env_config.task_name,
    #     seed=algo_config.seed,
    #     episode_length=env_config.episode_length,
    #     action_repeat=env_config.action_repeat,
    #     image_size=env_config.image_size,
    #     frame_stack=env_config.frame_stack,
    #     mode="train",
    # )
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
        video_dir if algo_config.save_video else None, height=448, width=448
    )
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    contrastive_buffer_baseline = utils.ContrastBuffer.load(expert_config.buffer_path, algo_config)
    contrastive_buffer_agent = utils.ContrastBuffer.load(expert_config.buffer_path, algo_config)
    cropped_visual_obs_shape = (
        3 * env_config.frame_stack,
        algo_config.image_crop_size,
        algo_config.image_crop_size,
    )
    # teacher
    teacher = torch.load(expert_config.model_path)
    teacher.eval()

    # baseline
    if baseline_config.load_path != None:
        print("load baseline from", baseline_config.load_path)
        baseline = FeatBaselineBC.load_baseline(baseline_config.load_path, cropped_visual_obs_shape, env.action_space.shape, baseline_config)
        baseline.set_expert(teacher)
    else:
        # train
        baseline = FeatBaselineBC(
            agent_obs_shape=cropped_visual_obs_shape,
            action_shape=env.action_space.shape,
            agent_config=baseline_config,
        )
        baseline.set_expert(teacher)

        print("===============Training baseline===============")
        L = Logger(work_dir, use_wandb=algo_config.use_wandb)
        start_time = time.time()
        for step in range(algo_config.baseline_bc_steps + 1):
            L.log("train/baseline_bc_duration", time.time() - start_time, step)
            start_time = time.time()
            L.dump(step)

            baseline.update(contrastive_buffer_baseline, L, step)

    # train features
    print("===============Training baseline features===============")
    L = Logger(work_dir, use_wandb=algo_config.use_wandb)
    for step in range(algo_config.baseline_feat_steps + 1):
        if step <= 100 and step % 10 == 0:
            print("Saving features")
            feature_show(baseline, contrastive_buffer_baseline, algo_config.feat_num, os.path.join(feature_dir, f"baseline_{step}.png"))

        baseline.update_crd_baseline(contrastive_buffer_baseline, L, step)

        L.dump(step)

    # agent
    if agent_config.load_path != None:
        agent = torch.load(agent_config.load_path)
        agent.set_expert(teacher)
        agent.train()
    else:
        agent = PureCrdBC(
            agent_obs_shape=cropped_visual_obs_shape,
            action_shape=env.action_space.shape,
            agent_config=agent_config,
        )
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
                feature_show(agent, contrastive_buffer_agent, algo_config.feat_num, os.path.join(feature_dir, f"agent_{step}.png"))

            agent.update(contrastive_buffer_agent, L, step)
        # save
        agent.clean_expert()
        torch.save(agent, os.path.join(model_dir, "pure_no_last.pt"))
        agent.set_expert(teacher)

    # show features
    print("===============Showing features===============")
    print("Baseline features")
    feat_baseline_s, feat_baseline_t = baseline.sample_feats(contrastive_buffer_baseline, n=algo_config.feat_num)
    np.savetxt(os.path.join(feature_dir, "baseline_s_example.txt"), feat_baseline_s[:10])
    np.savetxt(os.path.join(feature_dir, "baseline_t_example.txt"), feat_baseline_t[:10])
    print("Begin fitting")
    tsne = TSNE(n_components=2, random_state=0)
    feats_reduction = tsne.fit_transform(np.concatenate((feat_baseline_s, feat_baseline_t), axis=0))
    plt.scatter(feats_reduction[:algo_config.feat_num, 0], feats_reduction[:algo_config.feat_num, 1], c='r')
    plt.scatter(feats_reduction[algo_config.feat_num:, 0], feats_reduction[algo_config.feat_num:, 1], c='b')
    plt.savefig(os.path.join(work_dir, "baseline_feats.png"))
    print("Baseline features done")

    print("Agent features")
    feat_agent_s, feat_agent_t = agent.sample_feats(contrastive_buffer_agent, n=algo_config.feat_num)
    np.savetxt(os.path.join(feature_dir, "agent_s_example.txt"), feat_agent_s[:10])
    np.savetxt(os.path.join(feature_dir, "agent_t_example.txt"), feat_agent_t[:10])
    print("Begin fitting")
    tsne = TSNE(n_components=2, random_state=0)
    feats_reduction = tsne.fit_transform(np.concatenate((feat_agent_s, feat_agent_t), axis=0))
    plt.scatter(feats_reduction[:algo_config.feat_num, 0], feats_reduction[:algo_config.feat_num, 1], c='r')
    plt.scatter(feats_reduction[algo_config.feat_num:, 0], feats_reduction[algo_config.feat_num:, 1], c='b')
    plt.savefig(os.path.join(work_dir, "agent_feats.png"))
    print("Agent features done")

    # agent last layer
    print("===============Training agent last layer===============")
    # freeze agent features
    agent.actor.freeze_feature_layers()
    L = Logger(work_dir, use_wandb=algo_config.use_wandb)
    start_time = time.time()
    for step in range(algo_config.agent_last_layer_steps + 1):
        L.log("train/pure_crd_last_duration", time.time() - start_time, step)
        start_time = time.time()
        L.dump(step)

        if step % algo_config.agent_eval_freq == 0:
            print("Evaluating:", work_dir)
            evaluate(
                env,
                agent,
                video,
                algo_config.eval_episodes,
                L,
                step,
            )
            L.dump(step)

        agent.update_last_layer(contrastive_buffer_agent, L, step)

    # final evaluate baseline performance
    print("===============Final evaluate baseline performance===============")
    baseline_reward = evaluate(
        env,
        baseline,
        VideoRecorder(None),
        algo_config.eval_episodes,
        None,
        None,
    )
    print("Baseline reward:", baseline_reward)

    # save models
    baseline.clean_expert()
    torch.save(baseline, os.path.join(model_dir, "baseline.pt"))
    agent.clean_expert()
    torch.save(agent, os.path.join(model_dir, "pure_crd.pt"))

    print("Done!")
