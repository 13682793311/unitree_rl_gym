# 测试环境的生成
import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.utils.terrain.perlin import TerrainPerlin
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

# 定义配置类来模拟TerrainPerlin所需的配置结构
class TerrainConfig:
    def __init__(self):
        # 基本地形参数
        self.terrain_length = 8.0  # 每个环境的长度
        self.terrain_width = 8.0   # 每个环境的宽度
        self.num_rows = 2          # 环境行数
        self.num_cols = 2          # 环境列数
        self.horizontal_scale = 0.01  # 横向分辨率
        self.vertical_scale = 0.005  # 纵向分辨率
        self.slope_treshold = 0.5    # 坡度阈值
        
        # 摩擦系数等物理参数
        self.static_friction = 0.5
        self.dynamic_friction = 0.5
        self.restitution = 0.0
        
        # Perlin噪声参数
        self.TerrainPerlin_kwargs = {
            'frequency': 10,
            'fractalOctaves': 2,
            'fractalLacunarity': 2.0,
            'fractalGain': 0.25,
            'zScale': 0.23
        }

def main():
    # 设置命令行参数
    custom_parameters = [
        {"name": "--frequency", "type": float, "default": 12.0, "help": "噪声频率"},
        {"name": "--octaves", "type": int, "default": 3, "help": "分形噪声层数"},
        {"name": "--z_scale", "type": float, "default": 0.25, "help": "高度缩放系数"},
        {"name": "--num_rows", "type": int, "default": 2, "help": "地形行数"},
        {"name": "--num_cols", "type": int, "default": 2, "help": "地形列数"},
        # 添加以下两个参数定义
        {"name": "--device_id", "type": int, "default": 0, "help": "GPU设备ID"},
        #{"name": "--sim_device", "type": str, "default": "cuda:0", "help": "仿真设备类型"},
    ]
    args = gymutil.parse_arguments(description="TerrainPerlin测试", custom_parameters=custom_parameters)
    
    # 初始化gym
    gym = gymapi.acquire_gym()
    
    # 设置仿真参数
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # PhysX参数
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    
    # 创建仿真实例
    device_id = args.device_id
    device = args.sim_device
    sim = gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("*** 创建仿真失败")
        quit()
    
    # 创建地形配置
    terrain_cfg = TerrainConfig()
    # terrain_cfg.TerrainPerlin_kwargs["frequency"] = args.frequency
    # terrain_cfg.TerrainPerlin_kwargs["fractalOctaves"] = args.octaves
    # terrain_cfg.TerrainPerlin_kwargs["zScale"] = args.z_scale
    # terrain_cfg.num_rows = args.num_rows
    # terrain_cfg.num_cols = args.num_cols
    
    # 创建随机地形
    print("正在创建Perlin噪声地形...")
    terrain = TerrainPerlin(terrain_cfg, num_envs=1)
    
    # 添加地形到仿真
    print("正在添加地形到仿真...")
    terrain.add_terrain_to_sim(gym, sim, device=args.sim_device)
    
    # 创建观察者
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1920
    cam_props.height = 1080
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        print("*** 创建观察者失败")
        quit()
    
    # 设置相机位置
    total_length = terrain_cfg.terrain_length * args.num_rows
    total_width = terrain_cfg.terrain_width * args.num_cols
    cam_pos = gymapi.Vec3(total_length/2 - total_length/4, -total_width/2, total_length/3)
    cam_target = gymapi.Vec3(total_length/2, total_width/2, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    
    # 创建各种形状的物体
    asset_options = gymapi.AssetOptions()
    asset_options.density = 100.0
    
    ball_asset = gym.create_sphere(sim, 0.2, asset_options)
    box_asset = gym.create_box(sim, 0.4, 0.4, 0.4, asset_options)
    capsule_asset = gym.create_capsule(sim, 0.2, 0.4, asset_options)
    
    # 创建环境
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    
    print(f"创建环境网格: {args.num_rows} x {args.num_cols}")
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # 在每个环境位置放置一个随机物体
    objects = []
    for i in range(args.num_rows):
        for j in range(args.num_cols):
            # 计算位置
            x = (i + 0.5) * terrain_cfg.terrain_length
            y = (j + 0.5) * terrain_cfg.terrain_width
            
            # 获取该位置的地形高度
            px = int(x / terrain_cfg.horizontal_scale)
            py = int(y / terrain_cfg.horizontal_scale)
            px = min(px, terrain.heightsamples.shape[0]-1)
            py = min(py, terrain.heightsamples.shape[1]-1)
            terrain_height = terrain.heightsamples[px, py] * terrain_cfg.vertical_scale
            
            # 选择随机形状
            shape_idx = np.random.randint(0, 3)
            if shape_idx == 0:
                asset = ball_asset
                name = f"ball_{i}_{j}"
            elif shape_idx == 1:
                asset = box_asset
                name = f"box_{i}_{j}"
            else:
                asset = capsule_asset
                name = f"capsule_{i}_{j}"
            
            # 设置位置和旋转
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(x, y, terrain_height + 1.0)
            pose.r = gymapi.Quat.from_euler_zyx(
                np.random.uniform(0, 3.14),
                np.random.uniform(0, 3.14),
                np.random.uniform(0, 3.14)
            )
            
            # 创建物体
            handle = gym.create_actor(env, asset, pose, name, 0, 0)
            
            # 设置随机颜色
            color = gymapi.Vec3(
                np.random.uniform(0, 1),
                np.random.uniform(0, 1),
                np.random.uniform(0, 1)
            )
            gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, color)
            
            objects.append((handle, x, y, terrain_height))
    
    print("仿真运行中...")
    print("按 'ESC' 退出")
    print("按 'R' 重置物体")
    
    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 步进仿真
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 检查用户输入
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "reset" and evt.value > 0:
                print("重置物体...")
                for obj in objects:
                    handle, x, y, height = obj
                    
                    # 创建新的位置和旋转
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(x, y, height + 1.0)
                    pose.r = gymapi.Quat.from_euler_zyx(
                        np.random.uniform(0, 3.14),
                        np.random.uniform(0, 3.14),
                        np.random.uniform(0, 3.14)
                    )
                    
                    # 重置状态
                    gym.set_rigid_body_state(env, gym.get_actor_rigid_body_index(env, handle, 0), pose, gymapi.STATE_POS)
        
        # 更新图形和渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # 等待，保持60FPS
        gym.sync_frame_time(sim)
    
    # 清理资源
    print("销毁观察者...")
    gym.destroy_viewer(viewer)
    
    print("销毁仿真...")
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
