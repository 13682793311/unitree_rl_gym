from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
class GO1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_envs = 2048 # 4096
        num_observations = 45      # 45
        num_privileged_obs = 235    # 48+11*17=235
        
        use_lin_vel = False
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "height_measurements",
        ]

    # 加入传感器类
    class sensor:
        class proprioception:
            obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"] # 传感器的观测
            latency_range = [0.005, 0.045] # [s]
            latency_resampling_time = 5.0 # [s]

    class terrain(LeggedRobotCfg.terrain):
        selected = 'Terrain' # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        
        # 地形的表示形式：trimesh是三角网络，heightfield是二维网络
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # 水平方向分辨率
        horizontal_scale = 0.1 # [m]
        # 垂直方向分辨率
        vertical_scale = 0.005 # [m]
        # 边界大小
        border_size = 25 # [m]
        # 是否使用课程
        curriculum = True
        # 静摩擦
        static_friction = 1.0
        # 滑动摩擦
        dynamic_friction = 1.0
        # 误差代偿
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # 初始化地形的状态等级
        max_init_terrain_level = 7 # starting curriculum state   # 增大地形难度
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 地形类别包括：平坦坡，崎岖坡，正台阶，负台阶，离散地形,踏脚石，间隙，坑洼地形
        # 各种类型所占比例
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        #terrain_proportions = [0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # class terrain(LeggedRobotCfg.terrain):
    #     selected = 'TerrainPerlin'
    #     mesh_type = None
    #     measure_heights = True
    #     # x: [-0.5, 1.5], y: [-0.5, 0.5] range for go2
    #     measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    #     measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    #     horizontal_scale = 0.025 # [m]
    #     vertical_scale = 0.005 # [m]
    #     border_size = 5 # [m]
    #     curriculum = True
    #     static_friction = 1.0
    #     dynamic_friction = 1.0
    #     restitution = 0.
    #     max_init_terrain_level = 5 # starting curriculum state
    #     terrain_length = 4. #4.
    #     terrain_width = 4.  #4.
    #     num_rows= 16 # number of terrain rows (levels)
    #     num_cols = 16 # number of terrain cols (types)
    #     slope_treshold = 1.

    #     TerrainPerlin_kwargs = dict(
    #         zScale= 0.07,
    #         frequency= 10,
    #     )    
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-2.0, 2.0] # min max [m/s]
            lin_vel_y = [-2.0, 2.0]   # min max [m/s]   # 增大速度的变化值
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

        # 加入关节限制
    #     sdk_dof_range = dict(
    #         Hip_max = 1.047,
    #         Hip_min = -1.047,
    #         Thigh_max = 2.966,
    #         Thigh_min = -0.663,
    #         Calf_max = -0.837,
    #         Calf_min = -2.721
    #     )
    
    # # 加入更多的终止条件
    # class termination:
    #     termination_terms = [
    #         "roll",
    #         "pitch",
    #     ]

    #     roll_kwargs = dict(
    #         threshold= 3.0, # [rad]
    #     )
    #     pitch_kwargs = dict(
    #         threshold= 3.0, # [rad] # for leap, jump
    #     )

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        init_base_pos_range = dict(
            x= [0.2, 0.6],
            y= [-0.25, 0.25],
        )
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        clearance_height_target = -0.0
        class scales( LeggedRobotCfg.rewards.scales ):
            
            termination = -0.0
            tracking_lin_vel = 1.0  # 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05      # 惩罚基座在xy轴上的角速度
            orientation = -2.0      # 保持基座水平
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  0.
            collision = -1.
            feet_stumble = -0.5       # 机器人绊脚
            action_rate = -0.02      # 动作变化率
            stand_still = -0.5       # 在没有命令的情况下保持静止
            feet_slip = -0.
            foot_clearance = -0.
    
    ############# 直立 ##################
    # class rewards:
    #     class scales:
    #         termination = -0.0
    #         tracking_lin_vel = 2.0      # 减小追踪线速度
    #         tracking_ang_vel = 1.5
    #         lin_vel_z = -0.0
    #         ang_vel_xy = -0.0
    #         orientation = -0.0
    #         torques = -0.0002
    #         dof_pos_limits = -10.0
    #         dof_vel = -0.
    #         dof_acc = -2.5e-7
    #         base_height = -0.
    #         feet_air_time = 0.0
    #         collision = -1.
    #         feet_stumble = -0.0 
    #         action_rate = -0.01
    #         stand_still = -0.
    #         handstand_feet_height_exp = 100.0    # 修改后无明显变化
    #         handstand_feet_on_air = 1.0
    #         handstand_feet_air_time = 1.0
    #         handstand_orientation_l2 = -1.0
            
    #     only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    #     tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    #     soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
    #     soft_dof_vel_limit = 1.
    #     soft_torque_limit = 1.
    #     base_height_target = 0.25
    #     max_contact_force = 100. # forces above this value are penalized
    ######################################
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            heading = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

            # body_height_cmd = 2.0
            # gait_phase_cmd = 1.0
            # gait_freq_cmd = 1.0
            # footswing_height_cmd = 0.15
            # body_pitch_cmd = 0.3
            # body_roll_cmd = 0.3
        
        clip_observations = 100.
        clip_actions = 100.

class GO1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic'
        max_iterations = 1500
        run_name = 'cmd_x_2'
        experiment_name = 'go1_climbsteps' # go1爬楼梯

         # Load and resume
        resume = False # 断点重训
        load_run = '/home/kami/unitree_rl_dataset/unitree_rl_gym/logs/go1_climbsteps/Apr10_20-48-47_cmd_x_2' # -1 = last run 
        checkpoint = '3000'  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
  
