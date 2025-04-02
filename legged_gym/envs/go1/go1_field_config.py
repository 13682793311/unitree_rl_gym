import numpy as np
import os
from os import path as osp
from legged_gym.utils.helpers import merge_dict
from legged_gym.envs.go1.go1_config import GO1RoughCfg,GO1RoughCfgPPO

go1_const_dof_range = dict(
    Hip_max= 1.047,
    Hip_min=-1.047,
    Thigh_max= 2.966,
    Thigh_min= -0.663,
    Calf_max= -0.837,
    Calf_min= -2.721,
)

go1_action_scale = 0.5

class Go1FieldCfg( GO1RoughCfg ):
    class env( GO1RoughCfg.env ):
        num_envs = 4096

    class init_state( GO1RoughCfg.init_state ):
        pos = [0., 0., 0.7]
        zero_actions = False

    class sensor( GO1RoughCfg.sensor ):
        class proprioception( GO1RoughCfg.sensor.proprioception ):
            delay_action_obs = False
            latency_range = [0.04-0.0025, 0.04+0.0075] # comment this if it is too hard to train.
            latency_resampling_time = 2.0 # [s]

        # 深度相机
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link

    class terrain( GO1RoughCfg.terrain ):
        mesh_type = "trimesh"
        selected = "BarrierTrack"
        max_init_terrain_level = 0
        num_rows = 20
        num_cols = 80
        border_size = 5
        slope_treshold = 20.
        # curriculum = True # for tilt, crawl, jump, leap
        curriculum = False # for walk
        horizontal_scale = 0.025 # [m]
        pad_unavailable_info = True

        BarrierTrack_kwargs = merge_dict(GO1RoughCfg.terrain.BarrierTrack_kwargs, dict(
            options= [
                # "jump",
                "crawl",
                # "tilt",
                # "leap",
            ], # each race track will permute all the options
            # randomize_obstacle_order= True,
            track_width= 1.6,
            track_block_length= 2., # the x-axis distance from the env origin point
            wall_thickness= (0.04, 0.2), # [m]
            wall_height= 0.0,
            jump= dict(
                height= (0.2, 0.6),
                depth= (0.1, 0.8), # size along the forward axis
                fake_offset= 0.0, # [m] an offset that make the robot easier to get into the obstacle
                jump_down_prob= 0., # probability of jumping down use it in non-virtual terrain
            ),
            crawl= dict(
                height= (0.25, 0.5),
                depth= (0.1, 0.6), # size along the forward axis
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.24, 0.32),
                depth= (0.4, 1.), # size along the forward axis
                opening_angle= 0.0, # [rad] an opening that make the robot easier to get into the obstacle
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.2, 1.0),
                depth= (0.4, 0.8),
                height= 0.2,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        ))

        TerrainPerlin_kwargs = dict(
            zScale= [0.08, 0.15],
            frequency= 10,
        )

    class commands( GO1RoughCfg.commands ):
        class ranges( GO1RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0., 0.]


    class control( GO1RoughCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 0.5}
        action_scale = go1_action_scale
        torque_limits = [20., 20., 25.] * 4
        computer_clip_torque = False
        motor_clip_torque = False

    class asset( GO1RoughCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        sdk_dof_range = go1_const_dof_range
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base", "imu"]
        foot_name = "foot"
        front_hip_names = ["FR_hip_joint", "FL_hip_joint"]
        rear_hip_names = ["RR_hip_joint", "RL_hip_joint"]

    class termination( GO1RoughCfg.termination ):
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track",
        ]
        roll_kwargs = dict(
            threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.15, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = False
    class domain_rand( GO1RoughCfg.domain_rand ):
        class com_range( GO1RoughCfg.domain_rand.com_range ):
            x = [-0.2, 0.2]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        init_base_pos_range = merge_dict(GO1RoughCfg.domain_rand.init_base_pos_range, dict(
            x= [0.05, 0.6],
        ))
        init_base_rot_range = dict(
            roll= [-0.75, 0.75],
            pitch= [-0.75, 0.75],
        )
        # init_base_vel_range = [-1.0, 1.0]
        init_base_vel_range = dict(
            x= [-0.2, 1.5],
            y= [-0.2, 0.2],
            z= [-0.2, 0.2],
            roll= [-1., 1.],
            pitch= [-1., 1.],
            yaw= [-1., 1.],
        )
        init_dof_vel_range = [-5, 5]

    class rewards( GO1RoughCfg.rewards ):
        class scales:
            tracking_ang_vel = 0.05
            tracking_world_vel = 3.
            # world_vel_l2norm = -2.
            # alive = 3.
            legs_energy_substeps = -2e-5
            # penalty for hardware safety
            exceed_dof_pos_limits = -8e-1
            exceed_torque_limits_l1norm = -8e-1
            # penalty for walking gait, probably no need
            lin_vel_z = -1.
            ang_vel_xy = -0.05
            orientation = -4.
            dof_acc = -2.5e-7
            collision = -10.
            action_rate = -0.1
            delta_torques = -1e-7
            torques = -1.e-5
            yaw_abs = -0.8
            lin_pos_y = -0.8
            hip_pos = -0.4
            dof_error = -0.04
        soft_dof_pos_limit = 0.8 # only in training walking
        max_contact_force = 200.0
        
    class normalization( GO1RoughCfg.normalization ):
        dof_pos_redundancy = 0.2
        clip_actions_method = "hard"
        clip_actions_low = []
        clip_actions_high = []
        for sdk_joint_name, sim_joint_name in zip(
            ["Hip", "Thigh", "Calf"] * 4,
            [ # in the order as simulation
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            ],
        ):
            clip_actions_low.append( (go1_const_dof_range[sdk_joint_name + "_min"] + dof_pos_redundancy - GO1RoughCfg.init_state.default_joint_angles[sim_joint_name]) / go1_action_scale )
            clip_actions_high.append( (go1_const_dof_range[sdk_joint_name + "_max"] - dof_pos_redundancy - GO1RoughCfg.init_state.default_joint_angles[sim_joint_name]) / go1_action_scale )
        del dof_pos_redundancy, sdk_joint_name, sim_joint_name

    class sim( GO1RoughCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.2, 0.31, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go1FieldCfgPPO( GO1RoughCfgPPO ):
    class algorithm( GO1RoughCfgPPO.algorithm ):
        entropy_coef = 0.0
        clip_min_std = 0.2

    class policy( GO1RoughCfgPPO.policy ):
        mu_activation = None # use action clip method by env

    class runner( GO1RoughCfgPPO.runner ):
        experiment_name = "field_go1"
        resume = False
        load_run = None
        
        run_name = "".join(["WalkForward",
        ("_pEnergySubsteps" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.legs_energy_substeps, trim= "-", exp_digits= 1) if getattr(Go1FieldCfg.rewards.scales, "legs_energy_substeps", 0.0) != 0.0 else ""),
        ("_rTrackVel" + np.format_float_scientific(Go1FieldCfg.rewards.scales.tracking_world_vel, precision=1, exp_digits=1, trim="-") if getattr(Go1FieldCfg.rewards.scales, "tracking_world_vel", 0.0) != 0.0 else ""),
        ("_pWorldVel" + np.format_float_scientific(-Go1FieldCfg.rewards.scales.world_vel_l2norm, precision=1, exp_digits=1, trim="-") if getattr(Go1FieldCfg.rewards.scales, "world_vel_l2norm", 0.0) != 0.0 else ""),
        ("_aScale{:d}{:d}{:d}".format(
                int(Go1FieldCfg.control.action_scale[0] * 10),
                int(Go1FieldCfg.control.action_scale[1] * 10),
                int(Go1FieldCfg.control.action_scale[2] * 10),
            ) if isinstance(Go1FieldCfg.control.action_scale, (tuple, list)) \
            else "_aScale{:.1f}".format(Go1FieldCfg.control.action_scale)
        ),
        ("_actionClip" + Go1FieldCfg.normalization.clip_actions_method if getattr(Go1FieldCfg.normalization, "clip_actions_method", None) is not None else ""),
        ("_from" + "_".join(load_run.split("/")[-1].split("_")[:2]) if resume else "_noResume"),
        ])
        max_iterations = 20000
        save_interval = 500
