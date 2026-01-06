# -*- coding: utf-8 -*-
"""
现实信道场景生成器 - 创造信道差异性来展示注水算法的优化效果

不通过人为降低信道增益，而是通过以下方式创造信道差异：
1. 更大的地理范围分布
2. 不同的高度分布
3. 遮挡环境的差异
4. 多径衰落的随机性
5. 不同的功率约束设置
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from env import IoTSystem

class RealisticChannelGenerator:
    def __init__(self, iot_system):
        self.system = iot_system
        
    def scenario_1_large_area_deployment(self, num_nodes=10):
        """
        场景1：大范围部署 - IoT节点分布在更大的地理范围内
        这会自然产生不同的距离，从而产生信道差异
        """
        print("=== 场景1: 大范围地理分布 ===")
        
        # 在1000m x 1000m范围内随机分布IoT节点
        iot_positions = np.random.uniform(0, 1000, (num_nodes, 3))
        iot_positions[:, 2] = 0  # 地面节点
        
        # UAV在中心区域，高度100m
        uav_position = np.array([500, 500, 100])
        
        return iot_positions, uav_position
    
    def scenario_2_clustered_with_outliers(self, num_nodes=10):
        """
        场景2：集群分布 + 离群节点
        大部分节点在UAV附近，少数节点距离较远
        """
        print("=== 场景2: 集群分布 + 离群节点 ===")
        
        num_close = int(0.7 * num_nodes)  # 70%的节点在附近
        num_far = num_nodes - num_close   # 30%的节点较远
        
        # 近距离节点：在UAV周围200m范围内
        close_positions = np.random.uniform(-200, 200, (num_close, 2))
        close_positions += np.array([350, 350])  # 中心化
        
        # 远距离节点：在500-800m范围内
        far_positions = np.random.uniform(500, 800, (num_far, 2))
        
        # 合并并设置高度
        iot_positions = np.zeros((num_nodes, 3))
        iot_positions[:num_close, :2] = close_positions
        iot_positions[num_close:, :2] = far_positions
        iot_positions[:, 2] = 0  # 地面节点
        
        uav_position = np.array([350, 350, 100])
        
        return iot_positions, uav_position
    
    def scenario_3_height_variation(self, num_nodes=10):
        """
        场景3：不同高度的IoT节点
        一些节点在建筑物上，一些在地面
        """
        print("=== 场景3: 高度变化场景 ===")
        
        # 随机分布在500m x 500m范围内
        iot_positions = np.random.uniform(100, 600, (num_nodes, 3))
        
        # 随机设置不同高度：0-50m（模拟建筑物高度差异）
        iot_positions[:, 2] = np.random.uniform(0, 50, num_nodes)
        
        uav_position = np.array([350, 350, 100])
        
        return iot_positions, uav_position
    
    def scenario_4_realistic_urban(self, num_nodes=10):
        """
        场景4：现实城市环境
        结合距离差异、高度差异和环境参数变化
        """
        print("=== 场景4: 现实城市环境 ===")
        
        # 城市网格布局，但有随机偏移
        grid_size = int(np.sqrt(num_nodes)) + 1
        positions = []
        
        for i in range(num_nodes):
            # 基础网格位置
            row = i // grid_size
            col = i % grid_size
            base_x = col * 200 + 200
            base_y = row * 200 + 200
            
            # 添加随机偏移（模拟真实部署的不规则性）
            offset_x = np.random.normal(0, 50)
            offset_y = np.random.normal(0, 50)
            
            # 随机高度（0-30m，模拟建筑物）
            height = np.random.exponential(10)  # 指数分布，大部分节点在低处
            height = min(height, 50)  # 限制最大高度
            
            positions.append([base_x + offset_x, base_y + offset_y, height])
        
        iot_positions = np.array(positions[:num_nodes])
        uav_position = np.array([400, 400, 120])  # 稍高的UAV
        
        return iot_positions, uav_position
    
    def scenario_5_power_constraint_variation(self, num_nodes=10):
        """
        场景5：不同的功率约束设置
        通过调整系统参数来创造优化空间
        """
        print("=== 场景5: 功率约束变化 ===")
        
        # 常规分布
        iot_positions = np.random.uniform(200, 600, (num_nodes, 3))
        iot_positions[:, 2] = 0
        uav_position = np.array([400, 400, 100])
        
        # 关键：设置更严格的功率约束
        original_p_max = self.system.p_max
        original_p_total = self.system.p_total
        
        # 降低单节点最大功率约束，这样注水算法就不能给所有节点分配相同功率
        self.system.p_max = 0.08  # 从0.4降到0.08
        self.system.p_total = 0.6  # 总功率也相应调整
        
        print(f"调整功率约束: p_max={self.system.p_max}W, p_total={self.system.p_total}W")
        
        return iot_positions, uav_position, original_p_max, original_p_total

def compare_scenarios_waterfilling():
    """
    比较不同场景下注水算法的优化效果
    """
    system = IoTSystem()
    generator = RealisticChannelGenerator(system)
    
    scenarios = [
        ("Large Area", generator.scenario_1_large_area_deployment),
        ("Clustered + Outliers", generator.scenario_2_clustered_with_outliers),
        ("Height Variation", generator.scenario_3_height_variation),
        ("Urban Environment", generator.scenario_4_realistic_urban),
    ]
    
    plt.figure(figsize=(15, 12))
    
    for idx, (name, scenario_func) in enumerate(scenarios):
        print(f"\n{'='*50}")
        print(f"测试场景: {name}")
        print(f"{'='*50}")
        
        # 生成场景
        iot_positions, uav_position = scenario_func()
        
        # 计算信道增益
        channel_gains, distances, path_losses = system.calculate_cluster_gains(
            iot_positions, uav_position
        )
        
        print(f"信道增益范围: {np.min(channel_gains):.2e} ~ {np.max(channel_gains):.2e}")
        print(f"信道增益比值: {np.max(channel_gains)/np.min(channel_gains):.2f}")
        print(f"距离范围: {np.min(distances):.1f}m ~ {np.max(distances):.1f}m")
        
        # 均匀功率分配
        uniform_powers = np.full(len(iot_positions), system.p_total / len(iot_positions))
        uniform_rate, _ = system.calculate_communication_rate(channel_gains, uniform_powers)
        
        # 注水算法功率分配
        from water_fill import waterfilling
        optimal_powers, water_level = waterfilling(
            channel_gains, system.p_total, system.sigma2
        )
        optimal_rate, _ = system.calculate_communication_rate(channel_gains, optimal_powers)
        
        # 计算改善程度
        improvement = (optimal_rate - uniform_rate) / uniform_rate * 100
        
        print(f"均匀分配速率: {uniform_rate/1e6:.3f} Mbps")
        print(f"注水算法速率: {optimal_rate/1e6:.3f} Mbps")
        print(f"性能提升: {improvement:.2f}%")
        
        # 绘制结果
        plt.subplot(2, 2, idx + 1)
        
        # 绘制节点位置
        plt.scatter(iot_positions[:, 0], iot_positions[:, 1], 
                   c=channel_gains, s=60, cmap='viridis', alpha=0.7)
        plt.scatter(uav_position[0], uav_position[1], 
                   c='red', s=200, marker='^', label='UAV')
        
        # 添加功率信息
        for i, (pos, power) in enumerate(zip(iot_positions, optimal_powers)):
            plt.annotate(f'{power:.3f}W', 
                        (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.colorbar(label='Channel Gain')
        plt.title(f'{name}\nImprovement: {improvement:.2f}%')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_channel_scenarios.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: realistic_channel_scenarios.png")

def test_power_constraint_scenario():
    """
    测试功率约束变化场景
    """
    system = IoTSystem()
    generator = RealisticChannelGenerator(system)
    
    print(f"\n{'='*60}")
    print("功率约束对注水算法效果的影响")
    print(f"{'='*60}")
    
    # 测试不同的功率约束设置
    constraint_settings = [
        (0.4, 1.0),   # 原始设置：宽松约束
        (0.15, 1.0),  # 中等约束
        (0.08, 0.6),  # 严格约束
        (0.05, 0.4),  # 很严格约束
    ]
    
    # 固定场景
    iot_positions = np.array([
        [200, 200, 0], [300, 250, 0], [450, 300, 0], 
        [600, 350, 0], [750, 400, 0], [400, 500, 0],
        [350, 150, 0], [500, 200, 0], [650, 250, 0], 
        [300, 400, 0]
    ])
    uav_position = np.array([400, 300, 100])
    
    results = []
    
    for p_max, p_total in constraint_settings:
        # 设置约束
        system.p_max = p_max
        system.p_total = p_total
        
        print(f"\n测试约束: p_max={p_max}W, p_total={p_total}W")
        
        # 计算信道增益
        channel_gains, _, _ = system.calculate_cluster_gains(iot_positions, uav_position)
        
        # 均匀分配
        uniform_powers = np.full(len(iot_positions), p_total / len(iot_positions))
        uniform_rate, _ = system.calculate_communication_rate(channel_gains, uniform_powers)
        
        # 注水算法
        from water_fill import waterfilling
        optimal_powers, _ = waterfilling(channel_gains, p_total, system.sigma2)
        optimal_rate, _ = system.calculate_communication_rate(channel_gains, optimal_powers)
        
        improvement = (optimal_rate - uniform_rate) / uniform_rate * 100
        
        results.append({
            'p_max': p_max,
            'p_total': p_total,
            'uniform_rate': uniform_rate,
            'optimal_rate': optimal_rate,
            'improvement': improvement,
            'optimal_powers': optimal_powers
        })
        
        print(f"  均匀分配: {uniform_rate/1e6:.3f} Mbps")
        print(f"  注水算法: {optimal_rate/1e6:.3f} Mbps")
        print(f"  性能提升: {improvement:.2f}%")
        
        # 检查功率分配的差异性
        power_std = np.std(optimal_powers)
        power_range = np.max(optimal_powers) - np.min(optimal_powers)
        print(f"  功率标准差: {power_std:.4f}W")
        print(f"  功率范围: {power_range:.4f}W")
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    # 性能提升对比
    plt.subplot(2, 2, 1)
    improvements = [r['improvement'] for r in results]
    labels = [f"p_max={r['p_max']}\np_total={r['p_total']}" for r in results]
    plt.bar(range(len(results)), improvements, color='skyblue', alpha=0.7)
    plt.xlabel('Power Constraint Settings')
    plt.ylabel('Performance Improvement (%)')
    plt.title('Waterfilling Performance vs Power Constraints')
    plt.xticks(range(len(results)), labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 功率分配对比
    plt.subplot(2, 2, 2)
    for i, result in enumerate(results):
        plt.plot(result['optimal_powers'], 
                label=f"p_max={result['p_max']}", 
                marker='o', alpha=0.7)
    plt.xlabel('IoT Node Index')
    plt.ylabel('Allocated Power (W)')
    plt.title('Power Allocation Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 速率对比
    plt.subplot(2, 2, 3)
    uniform_rates = [r['uniform_rate']/1e6 for r in results]
    optimal_rates = [r['optimal_rate']/1e6 for r in results]
    
    x = range(len(results))
    width = 0.35
    plt.bar([i - width/2 for i in x], uniform_rates, width, 
           label='Uniform', alpha=0.7, color='orange')
    plt.bar([i + width/2 for i in x], optimal_rates, width, 
           label='Waterfilling', alpha=0.7, color='green')
    
    plt.xlabel('Constraint Settings')
    plt.ylabel('Data Rate (Mbps)')
    plt.title('Rate Comparison')
    plt.xticks(x, [f"{r['p_max']}" for r in results])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 功率利用效率
    plt.subplot(2, 2, 4)
    efficiencies = [r['optimal_rate']/r['p_total']/1e6 for r in results]
    plt.plot(range(len(results)), efficiencies, 
            marker='s', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Constraint Settings')
    plt.ylabel('Rate per Total Power (Mbps/W)')
    plt.title('Power Efficiency')
    plt.xticks(range(len(results)), [f"{r['p_max']}" for r in results])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('power_constraint_analysis.png', dpi=300, bbox_inches='tight')
    print("图片已保存为: power_constraint_analysis.png")
    
    return results

if __name__ == "__main__":
    print("现实信道场景测试 - 展示注水算法的真实优化效果")
    print("="*60)
    
    # 测试不同地理场景
    compare_scenarios_waterfilling()
    
    # 测试功率约束影响
    test_power_constraint_scenario()
    
    print("\n总结建议:")
    print("1. 增大IoT节点的地理分布范围")
    print("2. 创造高度差异（建筑物、地形）")
    print("3. 适当调整功率约束参数")
    print("4. 使用更现实的城市部署模式")
    print("5. 考虑遮挡和多径效应的随机性")
