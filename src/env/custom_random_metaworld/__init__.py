import re
from collections import OrderedDict
import pickle
import numpy as np
from metaworld import Task

from .sawyer_xyz.sawyer_door_close_v2 import SawyerDoorCloseEnvV2
from .sawyer_xyz.sawyer_window_close_v2 import SawyerWindowCloseEnvV2
from .sawyer_xyz.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from .sawyer_xyz.sawyer_button_press_wall_v2 import SawyerButtonPressWallEnvV2
from .sawyer_xyz.sawyer_assembly_peg_v2 import SawyerNutAssemblyEnvV2
from .sawyer_xyz.sawyer_hammer_v2 import SawyerHammerEnvV2
from .sawyer_xyz.sawyer_coffee_button_v2 import SawyerCoffeeButtonEnvV2
from .sawyer_xyz.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from .sawyer_xyz.sawyer_reach_wall_v2 import SawyerReachWallEnvV2

ALL_V2_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", SawyerNutAssemblyEnvV2),
        ("button-press-wall-v2", SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", SawyerCoffeeButtonEnvV2),
        ("door-close-v2", SawyerDoorCloseEnvV2),
        ("drawer-open-v2", SawyerDrawerOpenEnvV2),
        ("hammer-v2", SawyerHammerEnvV2),
        ("pick-place-v2", SawyerPickPlaceEnvV2),
        ("reach-wall-v2", SawyerReachWallEnvV2),
        ("window-close-v2", SawyerWindowCloseEnvV2),
    )
)

_ARGS_KWARGS = {
    key: dict(
        args=[],
        kwargs={
            "task_id": list(ALL_V2_ENVIRONMENTS.keys()).index(key),
        },
    )
    for key, _ in ALL_V2_ENVIRONMENTS.items()
}

_OVERRIDE = dict(partially_observable=False)

def create_observable_goal_envs():
    observable_goal_envs = {}
    for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
        d = {}

        def initialize(env, seed=None, render_mode=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()

            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")

        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)

ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE = create_observable_goal_envs()

_N_GOALS = 60

def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))

def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
    all_tasks = {}
    for random_level in [0, 1, 2, 3, 4]:
        level_key = f'level_{random_level}'
        level_tasks = {}
        for env_name, args in args_kwargs.items():
            env_key = env_name
            tasks = []
            assert len(args["args"]) == 0
            env = classes[env_name](random_level=random_level)
            env._freeze_rand_vec = False
            env._set_task_called = True
            rand_vecs = []
            kwargs = args["kwargs"].copy()
            del kwargs["task_id"]
            env._set_task_inner(**kwargs)
            for _ in range(_N_GOALS):
                env.reset()
                rand_vecs.append(env._last_rand_vec)
            unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
            assert unique_task_rand_vecs.shape[0] == _N_GOALS, unique_task_rand_vecs.shape[
                0
            ]
            env.close()
            for rand_vec in rand_vecs:
                kwargs = args["kwargs"].copy()
                del kwargs["task_id"]
                kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
                kwargs.update(kwargs_override)
                tasks.append(_encode_task(env_name, kwargs))
            del env
            level_tasks[env_key] = tasks
        all_tasks[level_key] = level_tasks
    if seed is not None:
        np.random.set_state(st0)
    return all_tasks

ALL_TASKS = _make_tasks(
    ALL_V2_ENVIRONMENTS,
    _ARGS_KWARGS,
    _OVERRIDE,
)
