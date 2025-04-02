import struct

# 建立了keymap的类，用于存储遥控器的按键信息
class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15

# 建立了RemoteController类，用于接收遥控器的数据
class RemoteController:
    def __init__(self):
        # 左侧摇杆位置
        self.lx = 0
        self.ly = 0
        # 右侧摇杆位置
        self.rx = 0
        self.ry = 0
        # 按键
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        # struct用于将二进制数据解包成python对象，unpack函数的第一个参数是指定二进制数据的格式，第二个参数是二进制数据
        # i是int，f是float，d是double，c是bytes，s是bytes，字符串
        # keys对应的是16个按键是否按下
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]
