"""DM10 motion-tracking profile on the shared NumPy Manager-Based runtime.

Injected into UniLab by ``scripts/install_to_unilab.sh``. Registering the task
name here is the only code UniLab needs; everything else (rewards,
observations, terminations) comes from the inherited task config.
"""

from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

DM10_MOTION_TASKS = (
    "DM10MotionTracking",
    "DM10MotionTrackingFlashSAC",
)

for _task_name in DM10_MOTION_TASKS:
    registry.register_env_config(_task_name, ManagerBasedRlEnvCfg)
    registry.register_env(_task_name, make_manager_based_rl_env, sim_backend="mujoco")

__all__ = ["DM10_MOTION_TASKS"]
