import numpy as np

class IoTSystem:
    def __init__(self):
        self.K = 4
        self.B = 100e3
        self.p_total = 1.0
        self.p_max = 0.4
        self.sigma2 = 10**(-90/10) / 1000
        self.f = 2e9
        self.c = 3e8
        self.H = 100

        self.a = 9.613      # 9.613  5.0188    
        self.b = 0.158           # 0.158 0.3511
        self.eta_LoS = 1.0     # LoS额外损耗  1.0  
        self.eta_NLoS = 20.0    # NLoS额外损耗  20.0
        
        self.data = 8e6
        
    def calculate_distance(self, iot_pos, uav_pos) ->float:
        """计算3D距离 d_ji"""
        # norm_l2 = np.linalg.norm(vector)  等价于 √(3² + 4²) = 5.0
        horizontal_dist = np.linalg.norm(iot_pos[:2] - uav_pos[:2])
        # 返回空间距离
        return np.sqrt(horizontal_dist**2 + self.H**2)
    
    def calculate_elevation_angle(self, iot_pos, uav_pos):
        """计算仰角 φ_ji (度)"""
        horizontal_dist = np.linalg.norm(iot_pos[:2] - uav_pos[:2])
        if horizontal_dist < 1e-6:  # 避免除零
            return 90.0
        φ_ji = np.arctan(self.H / horizontal_dist) * 180 / np.pi
        return φ_ji
    
    def calculate_path_loss_db(self, iot_pos, uav_pos) ->float:
        """
        计算路径损耗 δ_ji (dB)
        严格按照README.md公式18：
        δ_ji = 20*log10(4πf/c) + (η_LoS - η_NLoS)/(1+a*exp(-b*(φ_ji-a))) + η_NLoS
        """
        # 自由空间路径损耗
        fspl = 20 * np.log10(4 * np.pi * self.f / self.c)
        
        # 仰角相关的LoS/NLoS概率因子
        phi = self.calculate_elevation_angle(iot_pos, uav_pos)
        prob_factor = (self.eta_LoS - self.eta_NLoS) / (1 + self.a * np.exp(-self.b * (phi - self.a)))
        
        # 总路径损耗：严格按照公式18，没有额外的距离项
        total_loss = fspl + prob_factor + self.eta_NLoS
        
        return total_loss
    
    def calculate_large_scale_gain(self, iot_pos, uav_pos):
        """
        计算大尺度信道增益 l_ji
        公式: l_ji = d_ji^(-2) * 10^(-δ_ji/10)
        """
        distance = self.calculate_distance(iot_pos, uav_pos)
        path_loss_db = self.calculate_path_loss_db(iot_pos, uav_pos)
        
        # 距离平方衰落 * 路径损耗(线性)
        return (distance**(-2)) * (10**(-path_loss_db/10))
    
    def calculate_cluster_gains(self, iot_positions, uav_position):
        """计算集群所有节点的信道增益"""
        I_j = len(iot_positions)            #1*10
        channel_gains = np.zeros(I_j)        #1*10
        distances = np.zeros(I_j)            #1*10
        path_losses = np.zeros(I_j)           #1*10
        
        for i in range(I_j):
            channel_gains[i] = self.calculate_large_scale_gain(
                iot_positions[i], uav_position
            )
            distances[i] = self.calculate_distance(
                iot_positions[i], uav_position
            )
            path_losses[i] = self.calculate_path_loss_db(
                iot_positions[i], uav_position
            )
        
        return channel_gains, distances, path_losses

    def calculate_communication_rate(self, channel_gains, powers):

        I_j = len(channel_gains)
        individual_rates = np.zeros(I_j)
        
        # Σlog₂(1 + SINR_ji)
        total_user_rate = 0.0
        for i, (l_ji, p_ji) in enumerate(zip(channel_gains, powers)):
            if p_ji > 1e-15 and l_ji > 1e-20:
                # SINR计算：K*l_ji*p_ji/σ₀²
                sinr = (self.K * l_ji * p_ji) / (self.sigma2)
                user_rate_per_hz = np.log2(1 + sinr)
                user_rate_bps = self.B * user_rate_per_hz
                
                individual_rates[i] = user_rate_bps
                total_user_rate += user_rate_per_hz
                
                print(f"  节点{i+1:2d}: SINR={sinr:8.2f}, 速率={user_rate_bps/1e6:8.2f} Mbps")
            else:
                print(f"  节点{i+1:2d}: 功率为0，无传输")
        
        
        # 总速率 = 用户速率（已经包含带宽B）
        total_rate_bps = sum(individual_rates)  # 直接求和，避免重复乘B
        
        print(f"\n📈 速率汇总:")
        print(f"  用户速率项: {total_user_rate:.2f} Hz·log₂")
        print(f"  总系统速率: {total_rate_bps / 1e6:.3f} Mbps")
        
        return total_rate_bps, individual_rates


