import os
import hydra
import wandb
import argparse
from algorithms.train_vanilla_sac import train as train_sac
from algorithms.train_vanilla_bc import train as train_bc
from algorithms.train_aac import train as train_aac
from algorithms.train_mopa import train as train_mopa
from algorithms.train_crd_bc import train as train_crd_bc
from algorithms.train_pure_crd import train as train_pure_crd
# from algorithms.train_pieg_origin import Workspace as W
from omegaconf import OmegaConf

os.environ["MUJOCO_GL"] = "egl"

@hydra.main(
    version_base=None, config_path="../configs", config_name="vanilla_visual_config.yaml"
)
def main(cfg):
    if cfg.algo.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project="DFS",
            name=cfg.algorithm + "_" + str(cfg.algo.seed),
            # mode="offline", 
        ) # e5715e3e5cbb09c9a47ea3fec24fdb0dd0fd9aa1
    if cfg.algorithm == "vanilla_state_sac" or cfg.algorithm == "vanilla_visual_sac" or cfg.algorithm == "noisy_state_sac":
        train_sac(cfg)
    if cfg.algorithm == "vanilla_bc" or cfg.algorithm == "vanilla_bc_strong_aug" or cfg.algorithm == "vanilla_bc_aug_contrast":
        train_bc(cfg)
    if cfg.algorithm == "vanilla_aac":
        train_aac(cfg)
    if cfg.algorithm == "mopa":
        train_mopa(cfg)
    if cfg.algorithm == "crd_bc" or cfg.algorithm == "crd_bc_strong_aug" or cfg.algorithm == "crd_bc_aug_contrast":
        train_crd_bc(cfg)
    if cfg.algorithm == "pure_crd":
        train_pure_crd(cfg)
    # if cfg.algorithm == "pieg":
    #     from pathlib import Path
    #     root_dir = Path.cwd()
    #     workspace = W(cfg)
    #     snapshot = root_dir / 'snapshot.pt'
    #     if snapshot.exists():
    #         print(f'resuming: {snapshot}')
    #         workspace.load_snapshot()
    #     workspace.train()
    if cfg.algo.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()