import os
import hydra
import wandb
from algorithms.train_vanilla_sac import train
from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"


@hydra.main(
    version_base=None, config_path="../configs", config_name="vanilla_state_config"
)
def main(cfg):
    if cfg.algo.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project="DFS",
            name=cfg.algorithm + "_" + str(cfg.algo.seed),
            # mode="offline",
        )
    if cfg.algorithm == "vanilla_state_sac" or cfg.algorithm == "vanilla_visual_sac":
        train(cfg)
    if cfg.algo.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
