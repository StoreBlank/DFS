# DFS: Distillation From State-based Policy

A project under developing and exploration

## Structure

**configs**: config yaml files for starting an experiment

**Agent**: An entity that can act or learn in an environment, like a living 
creature. Parts of the agents (like organ) please put in `module.py`. Now have state-based sac, visual-based sac and vanilla bc agent

**Algorithm**: A blueprint introduces how to train an agent. Now have only vanilla state-based and visual-based sac

**Environment**: Openai gym-styled environments, have methods like `step()`, `reset()` and so on. Now have dm-control and distracting cs (mention that you need to download dm-control pkg: `pip install dm-control`)

**utils.py**: Other reusable tools and components are here, specially `ReplayBuffer` and `ContrastBuffer`

## TODO

**Agent**: 
1. Enable bc agent to update from representation, refer to https://github.com/HobbitLong/RepDistiller, you may first complete `utils.py`. Don't use `AliasMethod` in that repo, just uniformly sample. -- Done!

**Algorithm**:
1. (Ours) First get a state-based sac, then use DAgger (http://arxiv.org/abs/1011.0686) and contrastive representation distillation (crd) to train a visual-based actor. You can treat it as to use crd_loss + bc_loss in the update step of DAgger. -- distill actor but not critic (which still need state infomation in training)

**Others**:
1. maniskill2 env
2. crd in both critic and actor? so that we can continue training on just observation