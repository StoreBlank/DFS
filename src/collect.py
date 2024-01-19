import os
import hydra
import torch
from env.wrappers import make_env
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from env.wrapper_metaworld import wrap
import utils
from utils import collect_buffer
from datetime import datetime

os.environ["MUJOCO_GL"] = "egl"


@hydra.main(
    version_base=None, config_path="../configs", config_name="collect_visualbc_buffer_config"
)
def main(args):
    # parse config
    env_config = args.env
    agent_config = args.agent
    algo_config = args.algo

    # Set seed
    utils.set_seed_everywhere(algo_config.seed)

    # Create working directory
    work_dir = os.path.join(
        algo_config.log_dir,
        env_config.env_id,
        args.algorithm,
        str(algo_config.seed),
        str(datetime.now()),
    )
    print("Working directory:", work_dir)
    utils.make_dir(work_dir)

    # Prepare agent
    print(f"Loading expert weights from {agent_config.model_path} ...")
    agent = torch.load(agent_config.model_path)
    agent.eval()
    for param in agent.actor.parameters():
        param.requires_grad = False
    print("Expert loaded!")

    collect_buffer(agent, env_config, algo_config.rollout_steps, algo_config.batch_size, work_dir)


if __name__ == "__main__":
    main()
