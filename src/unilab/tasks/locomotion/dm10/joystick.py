"""Hydra-owned DM10 flat Manager-Based production registration.

DM10 = Damiao bipedal lower body, 10 DoF (leg_l1-5 + leg_r1-5) driven by
DM-J4340-2EC motors over CAN FD. Task behavior is declared in
``conf/ppo/task/dm10_joystick_flat/base.yaml``; this module only binds the
registry name to the generic Manager-Based runtime.
"""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("DM10JoystickFlat", ManagerBasedRlEnvCfg)
registry.register_env("DM10JoystickFlat", make_manager_based_rl_env, sim_backend="mujoco")
