# DFS: Distillation From State-based Policy

A project under developing and exploration

## Structure

**configs**: config yaml files for starting an experiment

**Agent**: An entity that can act or learn in an environment, like a living creature. Parts of the agents (like organ) please put in `module.py`.

**Algorithm**: A blueprint introduces how to train an agent.

**Environment**: Openai gym-styled environments, have methods like `step()`, `reset()` and so on.

**utils.py**: Other reusable tools and components are here, specially `ReplayBuffer` and `ContrastBuffer`

## Mention

1. You need to install the environment needed by Metaworld, and I remember that the mujoco version has conflict with previous environment. Please choose the version of Metaworld (no more usage of dm-control)

2. Metaworld has a bug: if you have two environment instances like env_1 = door-open-v2 and env_2 = window-open-v2, call env_1.reset() will destroy the assets in env_2. So please do not use reset() in multitask, instead directly creating another env instance.

3. Instances created from ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE are fixed, no randomization. Those from metaworld.MT10() will have randomization. Randomization comes from parameter `task` in env.

## TODO

More experiments in many possible directions.
