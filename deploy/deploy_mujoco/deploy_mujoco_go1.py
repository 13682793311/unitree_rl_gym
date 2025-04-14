import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


import pygame
from threading import Thread
# 加入手柄控制
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.0, 1.0, 1.0

joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = -joystick.get_axis(1) * x_vel_max
            y_vel_cmd = -joystick.get_axis(0) * y_vel_max
            yaw_vel_cmd = -joystick.get_axis(3) * yaw_vel_max

            pygame.time.delay(100)

        # launch gamepad thread
    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    FIX_COMMAND = False
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        # 观测限幅
        clip_observations = config["clip_observations"]

        # 动作限幅
        clip_actions = config["clip_actions"]

        # 臀关节限幅
        hip_scale_reduction = config["hip_scale_reduction"]

    # 模型输出(FL,FR,RL,RR)
    _action = np.zeros(num_actions, dtype=np.float32)
    
    # 下发给电机的目标关节角度(FR,FL,RR,RL)
    target_dof_pos = default_angles.copy()

    # 模型输入(FL,FR,RL,RR)
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            if FIX_COMMAND:
                cmd[0] = 0.5    # 1.0
                cmd[1] = 0.
                cmd[2] = 0.
                #cmd[3] = 0.
            else:
                cmd[0] = x_vel_cmd
                cmd[1] = y_vel_cmd
                cmd[2] = yaw_vel_cmd
                #cmd[3] = 0.
            # 低级控制器
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            #print("tau:",tau)
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # 将电机状态转换为模型输入
                _qj = [qj[3],qj[4],qj[5],
                       qj[0],qj[1],qj[2],
                       qj[9],qj[10],qj[11],
                       qj[6],qj[7],qj[8]]
                _dqj = [dqj[3],dqj[4],dqj[5],
                       dqj[0],dqj[1],dqj[2],
                       dqj[9],dqj[10],dqj[11],
                       dqj[6],dqj[7],dqj[8]]
                
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = _qj
                obs[9 + num_actions : 9 + 2 * num_actions] = _dqj
                
                # 上一次的模型输出
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = _action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # 对观测进行限幅
                obs_tensor = torch.clamp(obs_tensor,-clip_observations,clip_observations)
                # policy inference
                # 推理得到动作
                _action = policy(obs_tensor).detach().numpy().squeeze()
                
                # 动作限幅
                _action = np.clip(_action,-clip_actions,clip_actions)

                action_scaled = _action * action_scale
                indices=[0,3,6,9]
                # 单独对臀部进行缩放
                for i in indices:
                    action_scaled[i] *= hip_scale_reduction
                
                # 将action下发到各个关节
                action = np.array([action_scaled[3],action_scaled[4],action_scaled[5],
                       action_scaled[0],action_scaled[1],action_scaled[2],
                       action_scaled[9],action_scaled[10],action_scaled[11],
                       action_scaled[6],action_scaled[7],action_scaled[8]])
                # transform action to target_dof_pos
                target_dof_pos = action + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
