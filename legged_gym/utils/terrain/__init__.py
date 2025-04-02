import importlib      # 动态加载模块

# 地形类注册表，使用字典建立简化名称与完整导入路径之间的映射关系
terrain_registry = dict(
    Terrain= "legged_gym.utils.terrain.terrain:Terrain",
    BarrierTrack= "legged_gym.utils.terrain.barrier_track:BarrierTrack",
    TerrainPerlin= "legged_gym.utils.terrain.perlin:TerrainPerlin",
)
# 地形类加载函数
def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
