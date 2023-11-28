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
1. Enable bc agent to update from representation, refer to https://github.com/HobbitLong/RepDistiller, you may first complete `utils.py`. Don't use `AliasMethod` in that repo, just uniformly sample. -- wait for debugging in BC ...
2. Add an asymmetrical sac, refer to http://arxiv.org/abs/1710.06542. Mention that it need to be able to be initialized from external actor and critic. -- Done, check aac agent

**Algorithm**:
1. Vanilla bc. Done check bc_agent
2. https://proceedings.mlr.press/v164/liu22b.html. First get a state-based sac, then run bc to get a visual-based actor, then run aac (asymmetrical actor-critic) to make the actor on-policy. I think it should be a strong baseline. -- Done, the aac and bc can use pretrained model and self defined buffer to train, the config is in auged_aac_config.yaml
3. (Ours) First get a state-based sac, then use DAgger (http://arxiv.org/abs/1011.0686) and contrastive representation distillation (crd) to train a visual-based actor. You can treat it as to use crd_loss + bc_loss in the update step of DAgger. -- wait for debugging in BC ...

**Others**:
Any further good ideas, or bring in more simulation environment (like maniskill). Dog may want last consideration.  -- doing mani env in mopa-RL