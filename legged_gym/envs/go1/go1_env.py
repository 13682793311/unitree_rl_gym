from legged_gym.envs.base.legged_robot import LeggedRobot
import torch
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import os

from torch import Tensor
from typing import Tuple, Dict

from legged_gym.utils.terrain.perlin import TerrainPerlin
from legged_gym.utils.terrain import get_terrain_cls
from legged_gym.utils.math import quat_apply_yaw
class Go1Robot(LeggedRobot):
    # 给观测加上噪声
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        
        return noise_vec
    
    # 在观测中加入高程图信息
    def compute_observations(self):
        self.obs_buf = torch.cat((  
                                    self.base_ang_vel  * self.obs_scales.ang_vel,           # 3
                                    self.projected_gravity,                                 # 3
                                    self.commands[:, :3] * self.commands_scale,             # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                    self.dof_vel * self.obs_scales.dof_vel,   #  12
                                    self.actions  # 12
                                    ),dim=-1)

        self.privileged_obs_buf = torch.cat((  
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3
                                    self.projected_gravity, # 3
                                    self.commands[:, :3] * self.commands_scale, # 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12
                                    self.dof_vel * self.obs_scales.dof_vel, # 12
                                    self.actions # 12
                                    ),dim=-1)
        # 在教师网络中加入高程图信息
        if self.cfg.terrain.measure_heights:
            #self.root_states[:, 2].unsqueeze(1) 的形状为 (num_envs, 1)，而 self.measured_heights 通常的形状是 (num_envs, num_height_points)
            #PyTorch 的广播机制会自动将形状为 (num_envs, 1) 的张量沿第二个维度扩展成 (num_envs, num_height_points)，从而使得两者能够做逐元素的相减操作。
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
        # add perceptive inputs if not blind
        # 普通观测中加入深度相机信息
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    # 使用分形噪声地形进行训练
    # 创建障碍地形
    def _create_terrain(self):
        if getattr(self.cfg.terrain, "selected", None) is None:
            self._create_ground_plane()
        else:
            terrain_cls = self.cfg.terrain.selected
            self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs)
            self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_terrain()
        self._create_envs()

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        return self.terrain.get_terrain_heights(points)