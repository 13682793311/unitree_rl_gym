import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # 课程
        if cfg.curriculum:
            self.curiculum()
        # 选择某种特定地形
        # elif cfg.selected:
        #      self.selected_terrain()
        # 随机生成
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        # 提取地形类型
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        # 循环遍历所有子地形块
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 将一维索引k转换为二维索引i,j
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            # 创建子地形
            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            # 将terrain_type转化为函数
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        # 创建基础地形对象
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        # 坡度随难度线性增加
        slope = difficulty * 0.4
        # 台阶高度基础值0.05，最大可增加0.18
        step_height = 0.05 + 0.18 * difficulty
        # 障碍物高度基础值0.05，最大可增加0.2
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        # 踏脚石尺寸随难度减小
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        # 踏脚石间距在简单难度更小
        stone_distance = 0.05 if difficulty==0 else 0.1
        # 间隙大小与难度成正比
        gap_size = 1. * difficulty
        # 坑洼深度与难度成正比
        pit_depth = 1. * difficulty
        # 使用choice值与预定义的比例阈值比较，选择不同类型的地形
        # 金字塔斜坡地形
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1   # 一半情况使用下坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        # 带随机噪声的金字塔斜坡
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        # 金字塔阶梯地形
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        # 离散障碍物地形
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        # 踏脚石地形
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        # 间隙地形
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        # 坑洼地形
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    # 创建各种地形表示形式
    def _create_heightfield(self, gym, sim, device):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gym.HeightFieldParams()
        hf_params.column_scale = self.cfg.horizontal_scale
        hf_params.row_scale = self.cfg.horizontal_scale
        hf_params.vertical_scale = self.cfg.vertical_scale
        hf_params.nbRows = self.tot_cols
        hf_params.nbColumns = self.tot_rows 
        hf_params.transform.p.x = -self.cfg.border_size 
        hf_params.transform.p.y = -self.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.static_friction
        hf_params.dynamic_friction = self.cfg.dynamic_friction
        hf_params.restitution = self.cfg.restitution

        self.gym.add_heightfield(sim, self.heightsamples, hf_params)
    
    
    def _create_trimesh(self, gym, sim, device):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.vertices.shape[0]
        tm_params.nb_triangles = self.triangles.shape[0]

        tm_params.transform.p.x = -self.cfg.border_size 
        tm_params.transform.p.y = -self.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        gym.add_triangle_mesh(sim, self.vertices.flatten(order='C'), self.triangles.flatten(order='C'), tm_params)
        self.heightsamples = torch.tensor(self.heightsamples).view(self.tot_rows, self.tot_cols).to(device)
    
    # 将地形导入到isaacgym中
    def add_terrain_to_sim(self, gym, sim, device):
        if self.type == "heightfield":
            self._create_heightfield(gym, sim, device)
        elif self.type == "trimesh":
            self._create_trimesh(gym, sim, device)
        else:
            raise NotImplementedError("Terrain type {} not implemented".format(self.type))
    # 获取各点的高度
    def get_terrain_heights(self, points):
        """ Return the z coordinate of the terrain where just below the given points. """
        num_robots = points.shape[0]
        points += self.cfg.border_size
        points = (points/self.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0]-2)
        py = torch.clip(py, 0, self.heightsamples.shape[1]-2)

        heights1 = self.heightsamples[px, py]
        heights2 = self.heightsamples[px+1, py]
        heights3 = self.heightsamples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(num_robots, -1) * self.cfg.vertical_scale

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
