import numpy as np


class UAV:
    """无人机基类 - 代表一个无人机实体"""
    
    def __init__(self, uav_id, initial_position, uav_params):
        """
        初始化无人机
        
        Args:
            uav_id: 无人机ID
            initial_position: 初始位置 [x, y, z]
            uav_params: 无人机参数字典，包含：
                - f_uav: 计算频率 (Hz)
                - P_uav: 传输功率 (W)
                - B: 带宽 (Hz)
                - v_uav_min: 最小速度 (m/s)
                - v_uav_max: 最大速度 (m/s)
                - m_uav: 质量 (kg)
                - C_uav: 1bit数据所需的CPU周期数
                - K_uav: 计算能耗系数
                等等
        """
        self.uav_id = uav_id
        self.position = np.array(initial_position, dtype=np.float32)
        
        # 无人机性能参数
        self.f_uav = uav_params.get('f_uav', 1.2 * 10 ** 9)  # 计算频率
        self.P_uav = uav_params.get('P_uav', 5)  # 传输功率
        self.B = uav_params.get('B', 10 * 10 ** 6)  # 带宽
        self.v_uav_min = uav_params.get('v_uav_min', 0)  # 最小速度
        self.v_uav_max = uav_params.get('v_uav_max', 60)  # 最大速度
        self.m_uav = uav_params.get('m_uav', 2)  # 质量
        self.C_uav = uav_params.get('C_uav', 200)  # 1bit数据所需的CPU周期数
        self.K_uav = uav_params.get('K_uav', 10 ** -28)  # 计算能耗系数
        
        # 飞行物理参数
        self.n = uav_params.get('n', 4)  # 旋翼数量
        self.rho = uav_params.get('rho', 1.225)  # 空气密度
        self.S_uav = uav_params.get('S_uav', 0.01)  # 参考面积
        self.g = uav_params.get('g', 9.8)  # 重力加速度
        self.c_r = uav_params.get('c_r', 0.012)  # 滚动摩擦系数
        self.c_t = uav_params.get('c_t', 0.302)  # 升力系数
        self.A_r = uav_params.get('A_r', 0.0314)
        self.s_r = uav_params.get('s_r', 0.0955)
        self.c_f = uav_params.get('c_f', 0.131)
        self.d_r = uav_params.get('d_r', 0.834)
        
        # 边界限制
        self.boundary = uav_params.get('boundary', {
            'x_min': 200, 'x_max': 1150,
            'y_min': 200, 'y_max': 1150,
            'z_min': 200, 'z_max': 500
        })
    
    def move(self, action):
        """
        根据动作移动无人机
        
        Args:
            action: 动作数组 [d_normalized, theta_h_normalized, theta_v_normalized, ...]
                   前三个元素是移动相关的动作（已归一化到[0,1]）
        
        Returns:
            success: 移动是否成功（是否在边界内）
            old_position: 移动前的位置
        """
        # 保存旧位置
        old_position = self.position.copy()
        
        # 解析动作
        d = self.v_uav_min + action[0] * (self.v_uav_max - self.v_uav_min)  # 移动距离
        theta_horizontal = action[1] * 2 * np.pi  # 水平偏转角，范围0-2pi
        theta_vertical = action[2] * np.pi  # 垂直偏转角，范围0-pi
        
        # 计算在三个维度上的位移
        self.position[0] += d * np.sin(theta_vertical) * np.cos(theta_horizontal)  # x轴位移
        self.position[1] += d * np.sin(theta_vertical) * np.sin(theta_horizontal)  # y轴位移
        self.position[2] += d * np.cos(theta_vertical)  # z轴位移
        
        # 检查边界
        if self.check_boundary():
            return True, old_position, d, theta_vertical
        else:
            # 超出边界，恢复到旧位置
            self.position = old_position
            return False, old_position, d, theta_vertical
    
    def check_boundary(self):
        """
        检查无人机是否在边界内
        
        Returns:
            bool: True表示在边界内，False表示超出边界
        """
        x, y, z = self.position
        return (self.boundary['x_min'] <= x <= self.boundary['x_max'] and
                self.boundary['y_min'] <= y <= self.boundary['y_max'] and
                self.boundary['z_min'] <= z <= self.boundary['z_max'])
    
    def get_position(self):
        """获取当前位置"""
        return self.position.copy()
    
    def set_position(self, position):
        """设置位置"""
        self.position = np.array(position, dtype=np.float32)
    
    def get_state(self):
        """获取无人机状态（用于强化学习的观察）"""
        return self.position.copy()
    
    def reset(self, initial_position):
        """重置无人机到初始位置"""
        self.position = np.array(initial_position, dtype=np.float32)


class FUAV(UAV):
    """固定翼无人机类 - 继承自UAV基类"""
    
    def __init__(self, uav_id, fixed_position, fuav_params):
        """
        初始化固定翼无人机
        
        Args:
            uav_id: 无人机ID
            fixed_position: 固定位置 [x, y, z]
            fuav_params: 固定翼无人机参数字典
        """
        # 调用父类构造函数
        super().__init__(uav_id, fixed_position, fuav_params)
        
        # 固定翼无人机特有参数
        self.f_fuav = fuav_params.get('f_fuav', 3 * 10 ** 9)  # 计算频率（更高）
        self.C_fuav = fuav_params.get('C_fuav', 300)  # 1bit数据所需的CPU周期数
        self.is_fixed = True  # 标记为固定位置
    
    def move(self, action):
        """
        固定翼无人机不移动，重写父类方法
        
        Returns:
            success: 始终返回True
            old_position: 当前位置（不变）
            d: 0（没有移动）
            theta_vertical: 0
        """
        return True, self.position.copy(), 0, 0
    
    def check_boundary(self):
        """固定翼无人机始终在边界内"""
        return True

