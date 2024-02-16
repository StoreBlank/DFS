import os
import hydra
import torch
import numpy as np
from tqdm import tqdm
import random
from env.wrappers import make_env
import metaworld
from env.custom_random_metaworld import ALL_V2_ENVIRONMENTS, ALL_TASKS
from env.wrapper_metaworld import wrap
import utils
from video import VideoRecorder
from datetime import datetime
from ipdb import set_trace

os.environ["MUJOCO_GL"] = "egl"

TASK_IDX = 0


@hydra.main(
    version_base=None, config_path="../configs", config_name="collect_visualbc_buffer_config"
)
def main(args):
    # parse config
    env_config = args.env
    agent_config = args.agent
    algo_config = args.algo
    if not hasattr(env_config, 'camera_id'):
        env_config.camera_id = 0

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Create working directory
    work_dir = os.path.join(
        algo_config.log_dir,
        'collect_buffer',
        env_config.env_id,
        args.algorithm,
        str(algo_config.seed),
        str(datetime.now()),
    )
    print("Working directory:", work_dir)
    utils.make_dir(work_dir)
    utils.write_info(args, os.path.join(work_dir, "info.log"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(
        video_dir if algo_config.save_video else None, height=448, width=448, camera_id=env_config.camera_id
    )

    # Prepare agent
    print(f"Loading expert weights from {agent_config.model_path} ...")
    agent = torch.load(agent_config.model_path)
    agent.eval()
    for param in agent.actor.parameters():
        param.requires_grad = False
    print("Expert loaded!")

    collect_buffer(agent, env_config, algo_config.rollout_steps, algo_config.batch_size, video, work_dir)


def collect_buffer(agent, env_config, rollout_steps, batch_size, video, work_dir):
    assert torch.cuda.is_available(), "must have cuda enabled"

    # Initialize environments
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
            diversity = env_config.diversity
            collect_tasks_idxs =  np.random.choice(np.arange(len(ALL_TASKS[f'level_{random_level}'][env_config.env_id])), diversity, replace=False)
            collect_tasks = [ALL_TASKS[f'level_{random_level}'][env_config.env_id][idx] for idx in collect_tasks_idxs]
            # rollout tasks in order
            def initialize_env(env_id):
                global TASK_IDX
                env_cls = ALL_V2_ENVIRONMENTS[env_id]
                env = env_cls(random_level=random_level)
                task = collect_tasks[TASK_IDX % diversity]
                TASK_IDX += 1
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

    env = initialize_env(env_config.env_id)
    replay_buffer = utils.ReplayBuffer(
        action_shape=env.action_space.shape,
        capacity=rollout_steps,
        batch_size=batch_size,
    )
    
    start_step, episode, episode_reward, done = 0, 0, 0, True
    for step in tqdm(range(start_step, rollout_steps + 1), desc="Rollout Progress"):
        if done:
            if episode >= 1:
                video.save(f"{env_config.env_id}_{episode}.mp4")

            env.close()
            env = initialize_env(env_config.env_id)
            obs = env.reset()
            video.init(enabled=True)
            done = False
            print(f"Episode {episode} reward: {episode_reward}")
            episode_reward = 0
            episode += 1

        with torch.no_grad():
            mu, pi, log_std = agent.exhibit_behavior(obs)

        # Take step
        next_obs, reward, done, _ = env.step(pi)
        video.record(env)
        replay_buffer.add_behavior(obs, pi, mu, log_std, reward, next_obs, done)
        episode_reward += reward
        obs = next_obs

    if work_dir is not None:
        buffer_dir = utils.make_dir(work_dir)
        replay_buffer.save(os.path.join(buffer_dir, f"{rollout_steps}.pkl"))

    print("Completed rollout")
    print("Total episodes:", episode)
    env.close()
    return replay_buffer


if __name__ == "__main__":
    main()
