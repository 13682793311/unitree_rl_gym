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
        num_envs = 4096
        num_observations = 45      # no use,use obs_components
        num_privileged_obs = 235    # 48+11*17
        
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

    # 改写terrain类
    # class terrain:
    #     #selected = "TerrainPerlin" # TerrainPerlin,Terrain,BarrierTrack
    #     selected = None
    #     mesh_type = "trimesh" # "heightfield" # none, plane, heightfield or trimesh
    #     curriculum = True
    #     measure_heights = True   # 狗周围一部分区域的高程图
    #     measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
    #     measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
    #     horizontal_scale = 0.025 # [m]
    #     vertical_scale = 0.005 # [m]
    #     border_size = 5 # [m]
    #     curriculum = False
    #     static_friction = 1.0
    #     dynamic_friction = 1.0
    #     restitution = 0.
    #     max_init_terrain_level = 5 # starting curriculum state
    #     terrain_length = 4.
    #     terrain_width = 4.
    #     num_rows= 16 # number of terrain rows (levels)
    #     num_cols = 16 # number of terrain cols (types)
    #     slope_treshold = 1.
    #     # 障碍物的配置
    #     BarrierTrack_kwargs = dict(
    #         options= [
    #             # "jump",
    #             # "crawl",
    #             # "tilt",
    #             # "leap",
    #         ], # each race track will permute all the options
    #         track_width= 1.6,
    #         track_block_length= 2., # the x-axis distance from the env origin point
    #         wall_thickness= (0.04, 0.2), # [m]
    #         wall_height= -0.05,
    #         jump= dict(
    #             height= (0.2, 0.6),
    #             depth= (0.1, 0.8), # size along the forward axis
    #             fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
    #             jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
    #         ),
    #         crawl= dict(
    #             height= (0.25, 0.5),
    #             depth= (0.1, 0.6), # size along the forward axis
    #             wall_height= 0.6,
    #             no_perlin_at_obstacle= False,
    #         ),
    #         tilt= dict(
    #             width= (0.24, 0.32),
    #             depth= (0.4, 1.), # size along the forward axis
    #             opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
    #             wall_height= 0.5,
    #         ),
    #         leap= dict(
    #             length= (0.2, 1.0),
    #             depth= (0.4, 0.8),
    #             height= 0.2,
    #         ),
    #         add_perlin_noise= True,
    #         border_perlin_noise= True,
    #         border_height= 0.,
    #         virtual_terrain= False,
    #         draw_virtual_terrain= True,
    #         engaging_next_threshold= 1.2,
    #         engaging_finish_threshold= 0.,
    #         curriculum_perlin= False,
    #         no_perlin_threshold= 0.1,
    #     )
    #     # 柏林噪声的配置
    #     TerrainPerlin_kwargs = dict(
    #         zScale= 0.07,
    #         frequency= 10,
    #     )
    class terrain(LeggedRobotCfg.terrain):
        selected = False # select a unique terrain type and pass all arguments
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
        max_init_terrain_level = 5 # starting curriculum state
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

        
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
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
        sdk_dof_range = dict(
            Hip_max = 1.047,
            Hip_min = -1.047,
            Thigh_max = 2.966,
            Thigh_min = -0.663,
            Calf_max = -0.837,
            Calf_min = -2.721
        )
    
    # 加入更多的终止条件
    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )

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
        clearance_height_target = -0.1
        class scales( LeggedRobotCfg.rewards.scales ):
            
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.
            feet_slip = -0.
            foot_clearance = -0.
    

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
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
        max_iterations = 800
        run_name = 'full'
        experiment_name = 'rough_go1'

  
