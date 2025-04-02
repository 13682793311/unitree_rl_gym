from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

#from legged_gym.envs.go1.go1_field_config import Go1FieldCfg,Go1FieldCfgPPO
#from legged_gym.envs.go1.go1_field_env import Go1Fieldenv
from legged_gym.envs.go1.go1_config import GO1RoughCfg, GO1RoughCfgPPO
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot
from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.envs.go1.go1_env import Go1Robot
from legged_gym.utils.task_registry import task_registry


task_registry.register( "go1", Go1Robot, GO1RoughCfg(), GO1RoughCfgPPO())
task_registry.register( "go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
# task_registry.register( "go1_field", Go1Fieldenv, Go1FieldCfg(), Go1FieldCfgPPO())