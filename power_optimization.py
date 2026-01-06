import numpy as np
from env import IoTSystem

class PowerOptimizer:
    """
    功率优化器：实现注水算法来优化IoT设备的传输功率
    """
    
    def __init__(self, iot_system):
        """
        初始化功率优化器
        
        Args:
            iot_system: IoTSystem实例
        """
        self.system = iot_system
        self.max_iterations = 100
        self.tolerance = 1e-8
        
    def waterfilling_power_allocation(self, channel_gains):
        """
        注水算法实现功率分配
        
        根据Shannon容量公式和拉格朗日乘数法：
        p_i* = (μ - σ²/(K*l_i))₊
        其中μ是水位，需要满足总功率约束：Σp_i = P_total
        
        Args:
            channel_gains: 各节点的信道增益数组
            
        Returns:
            optimized_powers: 优化后的功率分配
            mu: 最优水位
        """
        I_j = len(channel_gains)
        
        # 计算噪声功率与信道增益的比值（注水算法中的"噪声水位"）
        noise_levels = self.system.sigma2 / (self.system.K * channel_gains)
        
        print(f"🌊 注水算法开始...")
        print(f"   信道增益范围: {np.min(channel_gains):.2e} - {np.max(channel_gains):.2e}")
        print(f"   噪声水位范围: {np.min(noise_levels):.2e} - {np.max(noise_levels):.2e}")
        
        # 检查是否所有节点的单节点功率约束都不起作用
        # 如果p_max够大，那么注水算法就是标准的无约束注水
        max_possible_power = self.system.p_total + np.max(noise_levels)
        if self.system.p_max >= max_possible_power:
            print(f"   单节点功率约束不起作用 (p_max={self.system.p_max:.3f} >= {max_possible_power:.3f})")
            
            # 标准注水算法：所有节点最终获得相同的"水位+噪声"
            # μ - noise_level[i] = power[i]
            # Σpower[i] = P_total => Σ(μ - noise_level[i]) = P_total
            # μ*N - Σnoise_level[i] = P_total
            # μ = (P_total + Σnoise_level[i]) / N
            
            mu = (self.system.p_total + np.sum(noise_levels)) / I_j
            powers = mu - noise_levels
            
            # 确保功率非负
            powers = np.maximum(0, powers)
            
            # 重新调整μ以满足总功率约束（考虑非负约束）
            active_nodes = powers > 0
            if np.sum(active_nodes) > 0:
                mu = (self.system.p_total + np.sum(noise_levels[active_nodes])) / np.sum(active_nodes)
                powers[active_nodes] = mu - noise_levels[active_nodes]
                powers[~active_nodes] = 0
            
            print(f"   最优水位: {mu:.6e}")
            print(f"   活跃节点: {np.sum(active_nodes)}/{I_j}")
            
        else:
            # 有单节点功率约束的情况，使用二分搜索
            mu_min = 0
            mu_max = self.system.p_total + np.max(noise_levels)
            
            for iteration in range(self.max_iterations):
                mu = (mu_min + mu_max) / 2
                
                # 计算当前水位下的功率分配
                powers = np.maximum(0, mu - noise_levels)
                
                # 应用单节点功率约束
                powers = np.minimum(powers, self.system.p_max)
                
                # 检查总功率约束
                total_power = np.sum(powers)
                
                if abs(total_power - self.system.p_total) < self.tolerance:
                    print(f"   收敛！迭代次数: {iteration+1}, 水位: {mu:.6e}")
                    break
                elif total_power < self.system.p_total:
                    mu_min = mu  # 需要提高水位
                else:
                    mu_max = mu  # 需要降低水位
                    
            if iteration == self.max_iterations - 1:
                print(f"   达到最大迭代次数 {self.max_iterations}")
        
        # 最终检查和调整
        total_power = np.sum(powers)
        print(f"   分配的总功率: {total_power:.6f} W")
        print(f"   功率分配范围: {np.min(powers):.6f} - {np.max(powers):.6f} W")
        
        return powers, mu
    
    def optimize_power_allocation(self, iot_positions, uav_position, verbose=True):
        """
        完整的功率优化流程
        
        Args:
            iot_positions: IoT设备位置数组
            uav_position: UAV位置
            verbose: 是否显示详细信息
            
        Returns:
            results: 包含优化结果的字典
        """
        if verbose:
            print("🚀 开始功率优化流程...")
            print("=" * 60)
        
        # 步骤1: 计算信道增益
        if verbose:
            print("📡 步骤1: 计算信道增益")
        
        channel_gains, distances, path_losses = self.system.calculate_cluster_gains(
            iot_positions, uav_position
        )
        
        if verbose:
            print(f"   设备数量: {len(iot_positions)}")
            print(f"   距离范围: {np.min(distances):.1f} - {np.max(distances):.1f} m")
            print(f"   信道增益: {np.min(channel_gains):.2e} - {np.max(channel_gains):.2e}")
        
        # 步骤2: 注水算法优化功率
        if verbose:
            print("\n⚡ 步骤2: 注水算法功率优化")
        
        optimized_powers, water_level = self.waterfilling_power_allocation(channel_gains)
        
        if verbose:
            print(f"   总功率: {np.sum(optimized_powers):.3f} W")
            print(f"   功率范围: {np.min(optimized_powers):.3f} - {np.max(optimized_powers):.3f} W")
            print(f"   最优水位: {water_level:.6e}")
        
        # 步骤3: 计算优化后的通信速率
        if verbose:
            print("\n📊 步骤3: 计算优化后通信速率")
        
        total_rate_bps, individual_rates = self.system.calculate_communication_rate(
            channel_gains, optimized_powers
        )
        
        # 步骤4: 与等功率分配对比
        if verbose:
            print("\n🔄 步骤4: 与等功率分配对比")
        
        uniform_powers = np.full(len(iot_positions), self.system.p_total / len(iot_positions))
        uniform_rate_bps, uniform_individual_rates = self.system.calculate_communication_rate(
            channel_gains, uniform_powers
        )
        
        improvement = (total_rate_bps - uniform_rate_bps) / uniform_rate_bps * 100
        
        if verbose:
            print(f"   等功率分配速率: {uniform_rate_bps / 1e6:.3f} Mbps")
            print(f"   注水算法速率: {total_rate_bps / 1e6:.3f} Mbps")
            print(f"   性能提升: {improvement:.2f}%")
        
        # 整理结果
        results = {
            'channel_gains': channel_gains,
            'distances': distances,
            'path_losses': path_losses,
            'optimized_powers': optimized_powers,
            'water_level': water_level,
            'total_rate_bps': total_rate_bps,
            'individual_rates': individual_rates,
            'uniform_powers': uniform_powers,
            'uniform_rate_bps': uniform_rate_bps,
            'uniform_individual_rates': uniform_individual_rates,
            'improvement_percent': improvement
        }
        
        return results
    
    def analyze_power_distribution(self, results):
        """
        分析功率分配结果
        
        Args:
            results: optimize_power_allocation返回的结果字典
        """
        print("\n" + "=" * 60)
        print("📊 功率分配详细分析")
        print("=" * 60)
        
        channel_gains = results['channel_gains']
        distances = results['distances']
        optimized_powers = results['optimized_powers']
        individual_rates = results['individual_rates']
        
        print(f"{'节点':<4} {'距离(m)':<8} {'信道增益':<12} {'分配功率(W)':<12} {'速率(Mbps)':<10}")
        print("-" * 60)
        
        for i in range(len(optimized_powers)):
            print(f"{i+1:<4} {distances[i]:<8.1f} {channel_gains[i]:<12.2e} "
                  f"{optimized_powers[i]:<12.3f} {individual_rates[i]/1e6:<10.2f}")
        
        print("-" * 60)
        print(f"总计: 功率={np.sum(optimized_powers):.3f}W, "
              f"速率={np.sum(individual_rates)/1e6:.3f}Mbps")
        
        # 功率利用效率分析
        print(f"\n🔋 功率利用分析:")
        print(f"   平均功率: {np.mean(optimized_powers):.3f} W")
        print(f"   功率标准差: {np.std(optimized_powers):.3f} W")
        print(f"   功率利用率: {np.sum(optimized_powers)/self.system.p_total*100:.1f}%")
        
        # 找出性能最好和最差的节点
        best_node = np.argmax(individual_rates)
        worst_node = np.argmin(individual_rates)
        
        print(f"\n🏆 性能分析:")
        print(f"   最佳节点: 节点{best_node+1} (速率: {individual_rates[best_node]/1e6:.2f} Mbps)")
        print(f"   最差节点: 节点{worst_node+1} (速率: {individual_rates[worst_node]/1e6:.2f} Mbps)")
        print(f"   速率比值: {individual_rates[best_node]/individual_rates[worst_node]:.2f}:1")


def main():
    """
    测试功率优化算法
    """
    # 创建IoT系统
    system = IoTSystem()
    optimizer = PowerOptimizer(system)
    
    # 设置IoT设备位置和UAV位置
    iot_positions = np.array([
        [391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], 
        [352.51, 636.99, 0], [365.74, 971.82, 0], [320.80, 406.66, 0], 
        [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], 
        [267.70, 926.15, 0]
    ])
    uav_position = np.array([350, 350, 100])
    
    # 执行功率优化
    results = optimizer.optimize_power_allocation(iot_positions, uav_position)
    
    # 详细分析结果
    optimizer.analyze_power_distribution(results)


if __name__ == "__main__":
    main()
