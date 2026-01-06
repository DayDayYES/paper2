"""
解耦关联的全双工UAV通信环境
实现DUDe模式、全双工干扰模型、Gumbel-Softmax用户关联
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from .uav import UAV, FUAV


class UAVEnvDUDe:
    """解耦上下行关联的UAV环境类"""
    
    def __init__(self, c1, c2, num_uavs=2):
        """
        初始化环境
        
        Args:
            c1: 用户簇1的坐标列表
            c2: 用户簇2的坐标列表
            num_uavs: 旋翼无人机数量（默认2个）
        """
        # 用户设备相关
        self.cluster1 = c1
        self.cluster2 = c2
        self.uecord = self.cluster1 + self.cluster2
        self.ue_num = len(self.uecord)
        
        # 环境参数
        self.ground_width = 1200
        self.B = 10 * 10 ** 6  # 10MHz带宽
        self.B_back = 20 * 10 ** 6  # 回程链路带宽 20MHz
        self.sigma = 10 ** (-10)  # 噪声功率 -70dBm
        
        # 通信环境参数 (dense urban)
        self.c1 = 12.08  # eta_a
        self.c2 = 0.11   # eta_b
        self.beta_los = 1  # eta_los
        self.beta_nlos = 20  # eta_nlos
        self.f_c = 2 * 10 ** 9  # 载波频率 2GHz
        self.c = 3 * 10 ** 8  # 光速
        
        # 用户设备参数
        self.P_ue_max = 0.1  # UE最大发射功率 (W) - 固定
        self.P_ue = self.P_ue_max
        
        # 全双工自干扰抑制系数
        self.xi = 100  # 自干扰抑制能力 (dB)，越大抑制越好
        
        # 无人机参数
        uav_params = {
            'f_uav': 1.2 * 10 ** 9,
            'P_uav': 5,  # 最大发射功率
            'B': self.B,
            'v_uav_min': 0,
            'v_uav_max': 60,
            'm_uav': 2,
            'C_uav': 200,
            'K_uav': 10 ** -28,
            'n': 4,
            'rho': 1.225,
            'S_uav': 0.01,
            'g': 9.8,
            'c_r': 0.012,
            'c_t': 0.302,
            'A_r': 0.0314,
            's_r': 0.0955,
            'c_f': 0.131,
            'd_r': 0.834,
            'boundary': {
                'x_min': 200, 'x_max': 1150,
                'y_min': 200, 'y_max': 1150,
                'z_min': 200, 'z_max': 500
            }
        }
        
        # 固定翼无人机参数（作为MBS）
        fuav_params = uav_params.copy()
        fuav_params['f_fuav'] = 3 * 10 ** 9
        fuav_params['C_fuav'] = 300
        fuav_params['P_uav'] = 10  # MBS功率更大
        
        # 创建旋翼无人机
        self.uavs = []
        initial_positions = [
            [350, 350, 250],
            [850, 850, 250]
        ]
        for i in range(num_uavs):
            uav = UAV(uav_id=i, initial_position=initial_positions[i], uav_params=uav_params)
            self.uavs.append(uav)
        
        # 创建固定翼无人机（MBS，索引为0）
        self.mbs = FUAV(uav_id=0, fixed_position=[600, 600, 300], fuav_params=fuav_params)
        
        self.uav_num = len(self.uavs)
        self.total_bs_num = self.uav_num + 1  # 包括MBS
        
        # Gumbel-Softmax温度参数
        self.temperature = 1.0
        self.temp_min = 0.5
        self.temp_decay = 0.999
        
        # 权重系数
        self.w_rate = 1.0  # 速率权重
        self.w_power = 0.01  # 功率惩罚权重
        self.w_backhaul = 0.1  # 回程链路惩罚权重
        self.w_safety = 5.0  # 安全距离惩罚权重
        
        # 约束参数
        self.d_min = 100  # UAV间最小安全距离
        self.p_avg_max = 5.0  # 平均功率约束
        
        # 历史记录
        self.power_history = []  # 用于计算平均功率
        
        self.state = [uav.get_position() for uav in self.uavs]
    
    # ==================== 距离和路径损耗计算 ====================
    
    def calculate_all_distances(self):
        """
        计算所有节点间的距离
        
        Returns:
            d_bs_ue: 基站到UE的距离 [total_bs_num, ue_num]
            d_bs_bs: 基站之间的距离 [total_bs_num, total_bs_num]
        """
        # 获取所有基站位置（MBS + UAVs）
        all_bs_positions = [self.mbs.get_position()] + [uav.get_position() for uav in self.uavs]
        
        # 计算基站到UE的距离
        d_bs_ue = np.zeros((self.total_bs_num, self.ue_num))
        for i, bs_pos in enumerate(all_bs_positions):
            for j, ue_pos in enumerate(self.uecord):
                d_bs_ue[i, j] = np.linalg.norm(np.array(bs_pos) - np.array(ue_pos))
        
        # 计算基站之间的距离（用于计算基站间干扰）
        d_bs_bs = np.zeros((self.total_bs_num, self.total_bs_num))
        for i, bs_pos_i in enumerate(all_bs_positions):
            for j, bs_pos_j in enumerate(all_bs_positions):
                if i != j:
                    d_bs_bs[i, j] = np.linalg.norm(np.array(bs_pos_i) - np.array(bs_pos_j))
        
        return d_bs_ue, d_bs_bs
    
    def compute_path_loss(self, distance, height):
        """
        计算路径损耗（包括LoS/NLoS概率）
        
        Args:
            distance: 3D距离
            height: UAV高度
        
        Returns:
            PL: 平均路径损耗（线性值，非dB）
        """
        # 仰角，确保 arcsin 的输入在 [-1, 1] 范围内
        sin_value = np.clip(height / (distance + 1e-10), -1.0, 1.0)
        theta = np.arcsin(sin_value) * 180 / np.pi
        
        # LoS概率
        P_los = 1 / (1 + self.c1 * np.exp(-self.c2 * (theta - self.c1)))
        P_nlos = 1 - P_los
        
        # 自由空间路径损耗（线性值）
        alpha = (4 * np.pi * self.f_c * distance / self.c) ** 2
        
        # LoS和NLoS路径损耗
        PL_los = alpha * (10 ** (self.beta_los / 10))
        PL_nlos = alpha * (10 ** (self.beta_nlos / 10))
        
        # 平均路径损耗
        PL = P_los * PL_los + P_nlos * PL_nlos
        
        return PL
    
    def compute_all_path_loss(self, d_bs_ue, d_bs_bs):
        """
        计算所有链路的路径损耗
        
        Args:
            d_bs_ue: 基站到UE的距离 [total_bs_num, ue_num]
            d_bs_bs: 基站间距离 [total_bs_num, total_bs_num]
        
        Returns:
            PL_bs_ue: 基站到UE的路径损耗 [total_bs_num, ue_num]
            PL_bs_bs: 基站间路径损耗 [total_bs_num, total_bs_num]
        """
        # 获取所有基站的高度
        heights = [self.mbs.position[2]] + [uav.position[2] for uav in self.uavs]
        
        # 基站到UE的路径损耗
        PL_bs_ue = np.zeros((self.total_bs_num, self.ue_num))
        for i in range(self.total_bs_num):
            for j in range(self.ue_num):
                PL_bs_ue[i, j] = self.compute_path_loss(d_bs_ue[i, j], heights[i])
        
        # 基站间路径损耗（用于干扰计算）
        PL_bs_bs = np.zeros((self.total_bs_num, self.total_bs_num))
        for i in range(self.total_bs_num):
            for j in range(self.total_bs_num):
                if i != j:
                    # 基站间干扰，使用发射基站的高度
                    PL_bs_bs[i, j] = self.compute_path_loss(d_bs_bs[i, j], heights[j])
        
        return PL_bs_ue, PL_bs_bs
    
    # ==================== Gumbel-Softmax用户关联 ====================
    
    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        """
        Gumbel-Softmax采样
        
        Args:
            logits: 未归一化的对数概率 [N, M] (N个UE, M个基站)
            temperature: 温度参数
            hard: 是否使用hard模式（离散化）
        
        Returns:
            y: 软关联或硬关联 [N, M]
        """
        # 添加Gumbel噪声
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, logits.shape) + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        
        # Softmax
        y = np.exp(y / temperature)
        y = y / (np.sum(y, axis=1, keepdims=True) + 1e-10)
        
        if hard:
            # Hard模式：one-hot编码
            y_hard = np.zeros_like(y)
            y_hard[np.arange(len(y)), np.argmax(y, axis=1)] = 1
            # 保持梯度（在numpy中无法实现，但在PyTorch中是 y_hard - y.detach() + y）
            y = y_hard
        
        return y
    
    def parse_association_from_action(self, action_logits, hard=False):
        """
        从动作中解析用户关联
        
        Args:
            action_logits: 所有UAV输出的关联logits
                          每个UAV输出 [ue_num * 2] 维度（上行和下行）
            hard: 是否使用hard模式
        
        Returns:
            b_ul: 上行关联矩阵 [total_bs_num, ue_num]
            b_dl: 下行关联矩阵 [total_bs_num, ue_num]
        """
        # 初始化关联矩阵
        b_ul = np.zeros((self.total_bs_num, self.ue_num))
        b_dl = np.zeros((self.total_bs_num, self.ue_num))
        
        # 收集所有基站的logits
        # MBS始终参与关联选择
        all_logits_ul = []
        all_logits_dl = []
        
        # 每个UE都需要在所有基站中选择一个进行上行/下行
        # 动作格式：每个UAV输出 ue_num*2 个值（每个UE的上行和下行logit）
        
        # 重新组织logits: [ue_num, total_bs_num, 2]
        logits_matrix_ul = np.zeros((self.ue_num, self.total_bs_num))
        logits_matrix_dl = np.zeros((self.ue_num, self.total_bs_num))
        
        # MBS的logits设为0（作为基准）
        logits_matrix_ul[:, 0] = 0
        logits_matrix_dl[:, 0] = 0
        
        # 从每个UAV的动作中提取logits
        for uav_idx in range(self.uav_num):
            # action_logits[uav_idx] 形状应该是 [ue_num * 2]
            ue_logits = action_logits[uav_idx]
            
            # 前ue_num个是上行logits，后ue_num个是下行logits
            ul_logits = ue_logits[:self.ue_num]
            dl_logits = ue_logits[self.ue_num:]
            
            # 填入矩阵（UAV索引为uav_idx+1，因为0是MBS）
            logits_matrix_ul[:, uav_idx + 1] = ul_logits
            logits_matrix_dl[:, uav_idx + 1] = dl_logits
        
        # 应用Gumbel-Softmax
        b_ul_soft = self.gumbel_softmax(logits_matrix_ul, self.temperature, hard)
        b_dl_soft = self.gumbel_softmax(logits_matrix_dl, self.temperature, hard)
        
        # 转置回 [total_bs_num, ue_num]
        b_ul = b_ul_soft.T
        b_dl = b_dl_soft.T
        
        return b_ul, b_dl
    
    # ==================== 干扰和SINR计算（全双工） ====================
    
    def compute_interference_and_rate(self, b_ul, b_dl, p_bs, PL_bs_ue, PL_bs_bs):
        """
        计算全双工模式下的干扰和传输速率
        
        Args:
            b_ul: 上行关联矩阵 [total_bs_num, ue_num]
            b_dl: 下行关联矩阵 [total_bs_num, ue_num]
            p_bs: 基站发射功率 [total_bs_num]
            PL_bs_ue: 基站到UE的路径损耗 [total_bs_num, ue_num]
            PL_bs_bs: 基站间路径损耗 [total_bs_num, total_bs_num]
        
        Returns:
            rate_ul: 上行速率 [total_bs_num, ue_num]
            rate_dl: 下行速率 [total_bs_num, ue_num]
            I_ul: 上行干扰 [total_bs_num, ue_num]
            I_dl: 下行干扰 [total_bs_num, ue_num]
        """
        # 初始化
        I_ul = np.zeros((self.total_bs_num, self.ue_num))
        I_dl = np.zeros((self.total_bs_num, self.ue_num))
        rate_ul = np.zeros((self.total_bs_num, self.ue_num))
        rate_dl = np.zeros((self.total_bs_num, self.ue_num))
        
        # 计算上行干扰和速率（解耦关联 DUDe）
        for n in range(self.ue_num):
            # 找到UE n的上行关联基站u
            u = np.argmax(b_ul[:, n])
            
            # 上行干扰计算（基站u接收时的干扰）
            # I_ul_u = I_D_u,u + I_U_u,u + I_self_u
            
            # 1. 来自其他基站下行传输的干扰 I_D_u,u
            I_D_u_u = 0
            for v in range(self.total_bs_num):
                if v != u:
                    # 基站v的下行发射功率 * 路径损耗 * 是否有下行用户关联到v
                    for n_prime in range(self.ue_num):
                        I_D_u_u += b_dl[v, n_prime] * p_bs[v] / PL_bs_bs[v, u]
            
            # 2. 来自其他UE上行传输的干扰 I_U_u,u
            I_U_u_u = 0
            for n_prime in range(self.ue_num):
                if n_prime != n:
                    # 其他UE关联到同一基站u的上行干扰
                    I_U_u_u += b_ul[u, n_prime] * self.P_ue / PL_bs_ue[u, n_prime]
            
            # 3. 全双工自干扰 I_self_u
            I_self_u = p_bs[u] / (10 ** (self.xi / 10))
            
            # 总上行干扰
            I_ul[u, n] = I_D_u_u + I_U_u_u + I_self_u
            
            # 上行SINR和速率
            SINR_ul = (self.P_ue / PL_bs_ue[u, n]) / (I_ul[u, n] + self.sigma)
            rate_ul[u, n] = b_ul[u, n] * self.B * np.log2(1 + SINR_ul)
        
        # 计算下行干扰和速率
        for n in range(self.ue_num):
            # 找到UE n的下行关联基站v
            v = np.argmax(b_dl[:, n])
            
            # 下行干扰计算（UE n接收时的干扰）
            # I_dl_n = I_D_n,n + I_U_n,n
            
            # 1. 来自其他基站下行传输的干扰 I_D_n,n
            I_D_n_n = 0
            for v_prime in range(self.total_bs_num):
                if v_prime != v:
                    # 其他基站的下行干扰
                    for n_prime in range(self.ue_num):
                        if n_prime == n:  # 只考虑对当前UE的干扰
                            I_D_n_n += b_dl[v_prime, n_prime] * p_bs[v_prime] / PL_bs_ue[v_prime, n]
            
            # 2. 来自UE上行传输的干扰 I_U_n,n
            I_U_n_n = 0
            for n_prime in range(self.ue_num):
                if n_prime != n:
                    # 其他UE的上行干扰
                    # 计算UE n_prime到UE n的距离和路径损耗
                    d_ue_ue = np.linalg.norm(np.array(self.uecord[n]) - np.array(self.uecord[n_prime]))
                    PL_ue_ue = (4 * np.pi * self.f_c * d_ue_ue / self.c) ** 2 * (10 ** (self.beta_nlos / 10))
                    I_U_n_n += self.P_ue / PL_ue_ue
            
            # 总下行干扰
            I_dl[v, n] = I_D_n_n + I_U_n_n
            
            # 下行SINR和速率
            SINR_dl = (p_bs[v] / PL_bs_ue[v, n]) / (I_dl[v, n] + self.sigma)
            rate_dl[v, n] = b_dl[v, n] * self.B * np.log2(1 + SINR_dl)
        
        return rate_ul, rate_dl, I_ul, I_dl
    
    # ==================== 回程链路 ====================
    
    def compute_backhaul_rate(self, PL_bs_bs, p_bs):
        """
        计算UAV与MBS之间的回程链路速率
        
        Args:
            PL_bs_bs: 基站间路径损耗 [total_bs_num, total_bs_num]
            p_bs: 基站发射功率 [total_bs_num]
        
        Returns:
            backhaul_rate: 回程链路速率 [uav_num] (只有UAV需要回程)
        """
        backhaul_rate = np.zeros(self.uav_num)
        
        for m in range(self.uav_num):
            # UAV m到MBS(索引0)的回程链路
            # 简化：假设回程链路用单独频段，无干扰
            SINR_back = (p_bs[m + 1] / PL_bs_bs[m + 1, 0]) / self.sigma
            backhaul_rate[m] = self.B_back * np.log2(1 + SINR_back)
        
        return backhaul_rate
    
    # ==================== 飞行能耗 ====================
    
    def compute_fly_energy(self, d, theta_vertical, uav):
        """计算UAV飞行能耗"""
        a = 5
        theta_c = 0.5 * np.pi - theta_vertical
        F = np.sqrt(((uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.cos(theta_c)) ** 2 +
                    (-1 * (uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.sin(theta_c) - 
                     uav.m_uav * uav.g) ** 2) / uav.n
        
        term1 = (uav.c_r / 8) * (F / (uav.c_t * uav.A_r * uav.rho) + 3 * d**2) * \
                np.sqrt(F * uav.rho * uav.s_r ** 2 * uav.A_r / uav.c_t)
        term2 = (1 + uav.c_f) * F * np.sqrt(np.sqrt((F ** 2 / (4 * uav.rho ** 2 * uav.A_r ** 2)) + 
                                                      (d ** 4 / 4)) - d ** 2 / 2)
        term3 = 0.5 * uav.d_r * d ** 3 * uav.rho * uav.s_r * uav.A_r + \
                uav.m_uav * uav.g * d / uav.n * (np.sin(theta_c))
        P_fly = uav.n * (term1 + term2 + term3)
        
        return P_fly
    
    # ==================== 环境step ====================
    
    def step(self, actions):
        """
        环境执行一步
        
        Args:
            actions: 每个UAV的动作
                    action[i] = [v_norm, theta_h_norm, theta_v_norm, p_norm, ue0_ul_logit, ue0_dl_logit, ...]
                    维度: [3 + 1 + ue_num * 2]
        
        Returns:
            state: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息字典
        """
        # 归一化动作到[0,1]
        actions_normalized = [(a + 1) / 2 for a in actions]
        
        info = {}
        reward = 0
        fly_energy_list = []
        
        # 1. UAV移动和飞行能耗
        for i, uav in enumerate(self.uavs):
            # 解析移动动作
            v_norm, theta_h_norm, theta_v_norm = actions_normalized[i][:3]
            
            move_action = np.array([v_norm, theta_h_norm, theta_v_norm])
            success, old_pos, d, theta_v = uav.move(move_action)
            
            # 计算飞行能耗
            P_fly = self.compute_fly_energy(d, theta_v, uav)
            fly_energy_list.append(P_fly)
            
            # 边界惩罚
            if not success:
                reward -= 10
        
        # 2. 解析功率和用户关联
        p_bs = np.zeros(self.total_bs_num)
        p_bs[0] = self.mbs.P_uav  # MBS功率固定
        
        association_logits = []
        for i, uav in enumerate(self.uavs):
            # 解析功率（第4个元素）
            p_norm = actions_normalized[i][3]
            p_bs[i + 1] = p_norm * uav.P_uav  # 归一化到最大功率
            
            # 解析关联logits（从第5个元素开始）
            logits = actions[i][4:]  # 保持原始值作为logits
            association_logits.append(logits)
        
        # 记录功率历史
        self.power_history.append(p_bs[1:].copy())  # 只记录UAV的功率
        
        # 3. 计算距离和路径损耗
        d_bs_ue, d_bs_bs = self.calculate_all_distances()
        PL_bs_ue, PL_bs_bs = self.compute_all_path_loss(d_bs_ue, d_bs_bs)
        
        # 4. 用户关联（Gumbel-Softmax）
        b_ul, b_dl = self.parse_association_from_action(association_logits, hard=False)
        
        # 5. 计算干扰和速率
        rate_ul, rate_dl, I_ul, I_dl = self.compute_interference_and_rate(
            b_ul, b_dl, p_bs, PL_bs_ue, PL_bs_bs)
        
        # 6. 计算回程链路速率
        backhaul_rate = self.compute_backhaul_rate(PL_bs_bs, p_bs)
        
        # 7. 计算各项指标
        # 总上下行速率
        total_rate_ul = np.sum(rate_ul)
        total_rate_dl = np.sum(rate_dl)
        total_rate = total_rate_ul + total_rate_dl
        
        # 确保速率是有效值
        if not np.isfinite(total_rate):
            total_rate = 0.0
            total_rate_ul = 0.0
            total_rate_dl = 0.0
        
        # 总传输功率
        total_power = np.sum(p_bs) + np.sum(fly_energy_list)
        
        # 确保功率是有效值
        if not np.isfinite(total_power):
            total_power = 0.0
        
        # 回程链路约束违反惩罚
        backhaul_penalty = 0
        for m in range(self.uav_num):
            # UAV m的接入链路总速率
            access_rate_m = np.sum(rate_ul[m + 1, :]) + np.sum(rate_dl[m + 1, :])
            # 回程链路容量
            back_rate_m = backhaul_rate[m]
            # 如果违反约束
            if access_rate_m > back_rate_m:
                backhaul_penalty += (access_rate_m - back_rate_m) / 1e6  # 归一化
        
        # UAV间安全距离惩罚
        safety_penalty = 0
        for i in range(self.uav_num):
            for j in range(i + 1, self.uav_num):
                dist = np.linalg.norm(self.uavs[i].position - self.uavs[j].position)
                if dist < self.d_min:
                    safety_penalty += (self.d_min - dist) / self.d_min
        
        # 8. 计算奖励
        reward = (self.w_rate * total_rate / 1e6 -  # 归一化到Mbps
                  self.w_power * total_power -
                  self.w_backhaul * backhaul_penalty -
                  self.w_safety * safety_penalty)
        
        # 检查并处理 NaN 或无穷大
        if not np.isfinite(reward):
            reward = -100.0  # 返回一个惩罚值
            print(f"Warning: Invalid reward detected. total_rate={total_rate/1e6:.2f}, "
                  f"total_power={total_power:.2f}, backhaul_penalty={backhaul_penalty:.2f}")
        
        # 9. 温度退火
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
        
        # 10. 更新状态
        self.state = [uav.get_position() for uav in self.uavs]
        
        # 11. 信息记录
        info = {
            'total_rate': total_rate / 1e6,  # Mbps
            'rate_ul': total_rate_ul / 1e6,
            'rate_dl': total_rate_dl / 1e6,
            'total_power': total_power,
            'backhaul_penalty': backhaul_penalty,
            'safety_penalty': safety_penalty,
            'temperature': self.temperature,
            'avg_power': np.mean(self.power_history[-100:]) if len(self.power_history) > 0 else 0
        }
        
        return self.state, reward, False, info
    
    def reset(self):
        """重置环境"""
        # 重置UAV位置
        initial_positions = [
            [350, 350, 250],
            [850, 850, 250]
        ]
        for i, uav in enumerate(self.uavs):
            uav.reset(initial_positions[i])
        
        # 重置温度
        self.temperature = 1.0
        
        # 重置历史
        self.power_history = []
        
        # 更新状态
        self.state = [uav.get_position() for uav in self.uavs]
        
        return self.state
    
    def get_action_dim(self):
        """获取动作空间维度"""
        # [速度, 水平角, 垂直角, 功率] + [每个UE的上行和下行logit]
        return 4 + self.ue_num * 2
    
    def get_obs_dim(self):
        """获取观察空间维度"""
        # 每个UAV的位置（3维）
        return 3

