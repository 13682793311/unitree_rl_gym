from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
class GO1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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
        num_envs = 4096 # 4096
        
        # num_privileged_obs = 235    # 48+11*17=235
        
        use_lin_vel = False
        history_encoding = True
        # n_priv = 9  # 预估的线速度
        n_proprio = 45 # 普通观测
        history_len = 10  # 历史数据的长度
        n_priv_latent = 3 + 4 + 1 + 12 +12   # 显式优先观测量
        n_scan = 11 * 17
        num_observations = n_proprio + n_priv_latent + n_scan + history_len*n_proprio 
        
        
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
    

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
    
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]   # 增大速度的变化值
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


    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
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
        base_height_target = 0.24
        clearance_height_target = -0.20
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            
            termination = -0.0
            tracking_lin_vel = 1.0  
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05     
            orientation = -0.      
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.05      # 增加高度
            feet_air_time =  1.
            collision = -1.
            feet_stumble = -0.       
            action_rate = -0.01      
            stand_still = -0       
            feet_slip = -0.
            foot_clearance = -0.
            smoothness = -0.
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            heading = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        
        clip_observations = 100.
        clip_actions = 100.

class GO1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [16, 8]  # 128，64，32
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [16, 8]  # 64，20
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False 
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'MYActorCritic'
        algorithm_class_name = 'MYPPO'
        max_iterations = 2000
        run_name = 'base_height'
        experiment_name = 'go1_climbsteps' # go1爬楼梯

         # Load and resume
        resume = False # 断点重训
        load_run = '/home/kami/unitree_rl_dataset/unitree_rl_gym/logs/go1_climbsteps/Apr21_21-48-52_base_height' # -1 = last run 
        checkpoint = '800'  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
    
    # # 估计器的配置
    # class estimator:
    #     train_with_estimated_states = True
    #     learning_rate = 1.e-4
    #     hidden_dims = [128, 64]
    #     priv_states_dim = GO1RoughCfg.env.n_priv_latent
    #     num_prop = GO1RoughCfg.env.n_proprio
    #     # num_scan = LeggedRobotCfg.env.n_scan
    #     num_observations = GO1RoughCfg.env.num_observations