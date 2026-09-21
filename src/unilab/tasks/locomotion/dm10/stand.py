"""Hydra-owned DM10 standing (L0) Manager-Based registration.

DM10 = Damiao bipedal lower body, 10 DoF (leg_l1-5 + leg_r1-5) driven by
DM-J4340-2EC motors over CAN FD. Task behavior is declared in
``conf/ppo/task/dm10_stand/mujoco.yaml``; this module only binds the registry
name to the generic Manager-Based runtime.

Why a separate registry entry at all: ``resolve_reward_override_field`` looks the
task up by name before the env is built, so a task that exists as a YAML but not
here fails with ``Environment '<name>' is not registered`` — after CUDA init, so
it looks like a runtime fault rather than a missing registration.
"""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("DM10Stand", ManagerBasedRlEnvCfg)
registry.register_env("DM10Stand", make_manager_based_rl_env, sim_backend="mujoco")
