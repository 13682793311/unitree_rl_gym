from legged_gym.envs.base.legged_robot import LeggedRobot
import torch
class Go2Robot(LeggedRobot):
    
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
        #noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions 
    
        return noise_vec
    
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
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
