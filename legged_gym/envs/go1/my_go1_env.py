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
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from isaacgym.torch_utils import *
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
        
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[9:9+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3+9:6+9] = noise_scales.gravity * noise_level
        noise_vec[6+9:9+9] = 0. # commands
        noise_vec[9+9:9+self.num_actions+9] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions+9:9+2*self.num_actions+9] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions+9:9+3*self.num_actions+9] = 0. # previous actions
        
        # noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[3:6] = noise_scales.gravity * noise_level
        # noise_vec[6:9] = 0. # commands
        # noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

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
        priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                   0 * self.base_lin_vel,
                                   0 * self.base_lin_vel), dim=-1)
        
        # 将普通观测和特权观测拼接在一起
        self.obs_buf = torch.cat([priv_explicit,self.obs_buf], dim=-1)
        
        # 在教师网络中加入高程图信息
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
        
        
        
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
    
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if getattr(self.cfg.terrain, "selected", None) is not None:
            #assert getattr(self.cfg.terrain, "mesh_type", None) is None, "Cannot have both terrain.selected and terrain.mesh_type"
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = min(self.cfg.terrain.max_init_terrain_level, self.terrain.cfg.num_rows - 1)
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = getattr(self.cfg.env, "env_spacing", 3.)
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    
