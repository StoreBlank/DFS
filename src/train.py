import os
import hydra
import wandb
from algorithms.train_vanilla_sac import train as train_sac
from algorithms.train_vanilla_bc import train as train_bc
from algorithms.train_aac import train as train_aac
from algorithms.train_mopa import train as train_mopa
from algorithms.train_crd_bc import train as train_crd_bc
from algorithms.train_pure_crd import train as train_pure_crd
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
        ) # e5715e3e5cbb09c9a47ea3fec24fdb0dd0fd9aa1
    if cfg.algorithm == "vanilla_state_sac" or cfg.algorithm == "vanilla_visual_sac":
        train_sac(cfg)
    if cfg.algorithm == "vanilla_bc":
        train_bc(cfg)
    if cfg.algorithm == "vanilla_aac":
        train_aac(cfg)
    if cfg.algorithm == "mopa":
        train_mopa(cfg)
    if cfg.algorithm == "crd_bc":
        train_crd_bc(cfg)
    if cfg.algorithm == "pure_crd":
        train_pure_crd(cfg)
    if cfg.algo.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
