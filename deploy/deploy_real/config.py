from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml

# 读取yaml文件中的部署参数为deploy_real模块提供参数
class Config:
    def __init__(self, file_path) -> None:
        # 打开deploy_real/config.yaml文件
        with open(file_path, "r") as f:
            # 将yaml文件中的内容加载到config中
            config = yaml.load(f, Loader=yaml.FullLoader)
            # 控制周期，即多长时间向机器人发送一次指令
            self.control_dt = config["control_dt"]
            # 控制消息类型，有hg和go两种类型
            self.msg_type = config["msg_type"]
            # imu类型，有pelvis（骨盆）和torso（躯干）两种类型
            self.imu_type = config["imu_type"]
            # 该参数在h1机器人上独有，在msg_type为"go"类型下，设置了部分weak_motor序号下的电机的控制模式参数设置为1
			# 其他电机的控制模式参数设置为0x0A
			# 而msg_type为"hg"类型下全部电机的控制模式都设置为1
            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]
            # 底层电机指令话题名称
            self.lowcmd_topic = config["lowcmd_topic"]
            # 底层状态话题名称
            self.lowstate_topic = config["lowstate_topic"]
            # 网络pt文件的路径
            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            # 腿部关节到电机的映射
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            # 腿部关节的PD控制器参数
            self.kps = config["kps"]
            self.kds = config["kds"]
            # 腿部关节的默认角度
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            # 胳膊和腰部电机的映射
            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            # 胳膊和腰部电机的PD控制器参数
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            # 胳膊和腰部电机的默认角度
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            # 输入训练好的强化学习网络的观测值比例
            # 本体角速度的比例
            self.ang_vel_scale = config["ang_vel_scale"]
            # 关节位置比例
            self.dof_pos_scale = config["dof_pos_scale"]
            # 关节速度比例
            self.dof_vel_scale = config["dof_vel_scale"]
            # 动作比例
            self.action_scale = config["action_scale"]
            # 指令比例
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            # 指令最大值
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            # 动作的数量
            self.num_actions = config["num_actions"]
            # 观测值的数量
            self.num_obs = config["num_obs"]
