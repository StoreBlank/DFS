import os
import hydra

from algorithms.collect_visualbc_buffer import collect_buffer

os.environ["MUJOCO_GL"] = "egl"


@hydra.main(
    version_base=None, config_path="../configs", config_name="collect_visualbc_buffer_config"
)
def main(cfg):
    if cfg.algorithm == "visualbc_buffer_collector":
        collect_buffer(cfg)

if __name__ == "__main__":
    main()
