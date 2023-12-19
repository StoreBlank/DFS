from env.wrapper_dmc import make_env as make_dmc_env
from env.wrapper_maniskill import make_env as make_maniskill_env

def make_env(**kwargs):
    if kwargs['category'] == 'dmc':
        del kwargs['category']
        return make_dmc_env(**kwargs)
    elif kwargs['category'] == 'maniskill':
        del kwargs['category']
        return make_maniskill_env(**kwargs)
    else:
        raise NotImplementedError
