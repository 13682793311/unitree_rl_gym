from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        # 读入yaml中的参数，创建一个Config对象
        self.config = config
        # 初始化RemoteController对象
        self.remote_controller = RemoteController()

        # Initialize the policy network
        # 读取训练好的强化学习网络策略
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        # 初始化过程变量
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # 动作
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # 目标位置
        self.target_dof_pos = config.default_angles.copy()
        # 观测量数组
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # 指令，速度和角速度
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # 根据不同的消息类型设置下发底层指令和接收底层状态的函数，话题等
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            # g1和h1_2使用hg消息类型
            # 底层的指令和状态
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            # 电机模式
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            # 初始化底层指令下发的话题以及函数
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            # h1使用go消息类型
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        # 等待接收底层状态
        self.wait_for_low_state()

        # Initialize the command msg
        # 初始化命令消息
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        # 用自身定义的CRC()函数对下发的命令进行数据完整性进行验证
        cmd.crc = CRC().Crc(cmd)
        # 控制指令发送
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        # 规定到初始默认位姿的时间是2s 
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        # 将腿部，腰部和手臂的参数合并
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        # 记录当前每个关节的位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        # 控制关节移动到默认位置
        for i in range(num_step):
            # alpha是当前执行到目标位置的步数比例
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                # 通过线性插值的方式，将当前关节的位置从初始位置移动到目标位置
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        # 当前步数
        self.counter += 1
        # Get the current joint position and velocity
        # 获取当前腿部关节位置和速度
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        # imu的四元数状态为w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # h1和h1_2的imu在躯干上,需要将imu数据转换到骨盆坐标系
        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # 腰部的位置和速度
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            # 该函数定义在rotation_helper.py中，将imu数据转换到骨盆坐标系
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        # 建立强化学习网络策略需要的观测值
        # 该函数在rotation_helper.py中定义，获取重力方向
        gravity_orientation = get_gravity_orientation(quat)
        # 电机位置
        qj_obs = self.qj.copy()
        # 电机速度
        dqj_obs = self.dqj.copy()
        # 电机旋转角度归一化
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        # 电机角速度归一化
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        # 机器人角速度归一化
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        # 当前步数乘以控制周期=当前时间
        count = self.counter * self.config.control_dt
        # 正弦和余弦相位
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # 底层指令
        # 机器人遥控器的左摇杆y值代表给机器人的期望x轴线速度指令
        self.cmd[0] = self.remote_controller.ly
        # 机器人遥控器的左摇杆x值代表给机器人的期望y轴线速度指令
        self.cmd[1] = self.remote_controller.lx * -1
        # 机器人遥控器的右摇杆x值代表给机器人的期望角速度指
        self.cmd[2] = self.remote_controller.rx * -1

        # 观测值数组
        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        # 从策略网络中获取动作
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        # 位置控制，返回目标位置
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        # 下发腿部电机目标角度
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        # 下发胳膊和腰部电机目标角度
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        # 发送指令
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    # 调用解析命令行参数的标准模块
    import argparse
    # 创建解析器对象，它负责处理命令行参数的解析工作。当程序运行时，解析器会自动处理sys.argv中的命令行参数
    # 设置部署脚本命令行需要的参数，包括机器人的网卡名字以及配置文件的名字
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    # 加载config文件
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    # 初始化DDS通信，在unitree_sdk2py脚本中设置
    ChannelFactoryInitialize(0, args.net)

    # 初始化Controller对象
    controller = Controller(config)
 
    # Enter the zero torque state, press the start key to continue executing
    # 进入零力矩状态，按下start键继续执行
    controller.zero_torque_state()

    # Move to the default position
    # 各关节移动到默认位置
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    # 进入默认位置状态，按下A键继续执行
    controller.default_pos_state()

    while True:
        try:
            # 收集观测值，下达底层指令
            controller.run()
            # Press the select key to exit
            # 按下select键退出
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    # 创建一个阻尼指令，下达底层指令，该函数在common/command_helper.py中定义，就是将所有电机的D参数设置为8
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
