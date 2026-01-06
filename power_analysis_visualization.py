import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from env import IoTSystem
from power_optimization import PowerOptimizer
# 设置matplotlib参数以支持科研图表
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'figure.figsize': (15, 10),
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'legend.frameon': True,
    'legend.framealpha': 0.9
})

def create_power_analysis_plots():
    """
    创建功率分析可视化图表
    """
    # 创建系统和优化器
    system = IoTSystem()
    optimizer = PowerOptimizer(system)
    
    # IoT设备位置
    iot_positions = np.array([
        [391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], 
        [352.51, 636.99, 0], [365.74, 971.82, 0], [320.80, 406.66, 0], 
        [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], 
        [267.70, 926.15, 0]
    ])
    uav_position = np.array([350, 350, 100])
    
    # 获取优化结果
    results = optimizer.optimize_power_allocation(iot_positions, uav_position, verbose=False)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UAV-IoT System Power Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 数据准备
    channel_gains = results['channel_gains']
    distances = results['distances']
    optimized_powers = results['optimized_powers']
    uniform_powers = results['uniform_powers']
    individual_rates = results['individual_rates']
    uniform_individual_rates = results['uniform_individual_rates']
    
    node_indices = np.arange(1, len(iot_positions) + 1)
    
    # 子图1: 功率分配对比
    ax1 = axes[0, 0]
    width = 0.35
    x_pos = np.arange(len(node_indices))
    
    bars1 = ax1.bar(x_pos - width/2, optimized_powers, width, 
                    label='Waterfilling Algorithm', alpha=0.8, color='skyblue', edgecolor='navy')
    bars2 = ax1.bar(x_pos + width/2, uniform_powers, width,
                    label='Uniform Power Allocation', alpha=0.8, color='lightcoral', edgecolor='darkred')
    
    ax1.set_xlabel('IoT Node Index')
    ax1.set_ylabel('Allocated Power (W)')
    ax1.set_title('(a) Power Allocation Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(node_indices)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, (opt_p, uni_p) in enumerate(zip(optimized_powers, uniform_powers)):
        ax1.text(i - width/2, opt_p + 0.005, f'{opt_p:.3f}', 
                ha='center', va='bottom', fontsize=9, rotation=90)
        ax1.text(i + width/2, uni_p + 0.005, f'{uni_p:.3f}', 
                ha='center', va='bottom', fontsize=9, rotation=90)
    
    # 子图2: 信道增益与功率分配关系
    ax2 = axes[0, 1]
    scatter = ax2.scatter(channel_gains, optimized_powers, s=150, 
                         c=distances, cmap='viridis', alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Channel Gain')
    ax2.set_ylabel('Allocated Power (W)')
    ax2.set_title('(b) Channel Gain vs Power Allocation')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 添加节点编号
    for i, (gain, power) in enumerate(zip(channel_gains, optimized_powers)):
        ax2.annotate(f'{i+1}', (gain, power), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Distance (m)')
    
    # 子图3: 速率提升对比
    ax3 = axes[0, 2]
    rate_improvement = (individual_rates - uniform_individual_rates) / uniform_individual_rates * 100
    
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in rate_improvement]
    bars = ax3.bar(node_indices, rate_improvement, color=colors, alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('IoT Node Index')
    ax3.set_ylabel('Rate Improvement (%)')
    ax3.set_title('(c) Waterfilling Algorithm Rate Improvement')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 显示数值
    for i, improvement in enumerate(rate_improvement):
        ax3.text(i+1, improvement + 0.01 if improvement >= 0 else improvement - 0.01, 
                f'{improvement:.3f}%', ha='center', 
                va='bottom' if improvement >= 0 else 'top', fontsize=9)
    
    # 子图4: 噪声水位分析
    ax4 = axes[1, 0]
    noise_levels = system.sigma2 / (system.K * channel_gains)
    
    ax4.semilogy(node_indices, noise_levels, 'o-', linewidth=2, markersize=8, 
                label='Noise Level', color='orange')
    ax4.axhline(y=np.mean(noise_levels), color='red', linestyle='--', 
                linewidth=2, label=f'Average: {np.mean(noise_levels):.2e}')
    
    ax4.set_xlabel('IoT Node Index')
    ax4.set_ylabel('Noise Level (σ²/K*l_i)')
    ax4.set_title('(d) Node Noise Level Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 功率效率分析
    ax5 = axes[1, 1]
    
    # 计算功率效率 (bps/W)
    opt_efficiency = individual_rates / optimized_powers
    uni_efficiency = uniform_individual_rates / uniform_powers
    
    x_pos = np.arange(len(node_indices))
    bars1 = ax5.bar(x_pos - width/2, opt_efficiency/1e6, width, 
                    label='Waterfilling Algorithm', alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    bars2 = ax5.bar(x_pos + width/2, uni_efficiency/1e6, width,
                    label='Uniform Power Allocation', alpha=0.8, color='lightsalmon', edgecolor='darkred')
    
    ax5.set_xlabel('IoT Node Index')
    ax5.set_ylabel('Power Efficiency (Mbps/W)')
    ax5.set_title('(e) Power Efficiency Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(node_indices)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 子图6: 系统性能汇总
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 性能统计
    total_opt_rate = np.sum(individual_rates) / 1e6
    total_uni_rate = np.sum(uniform_individual_rates) / 1e6
    total_improvement = (total_opt_rate - total_uni_rate) / total_uni_rate * 100
    
    power_variance_opt = np.var(optimized_powers)
    power_variance_uni = np.var(uniform_powers)
    
    summary_text = f"""
    System Performance Summary
    ─────────────────────────
    
    Rate Performance:
    • Waterfilling Total Rate: {total_opt_rate:.3f} Mbps
    • Uniform Allocation Rate: {total_uni_rate:.3f} Mbps
    • Overall Improvement: {total_improvement:.4f}%
    
    Power Allocation:
    • Waterfilling Power Variance: {power_variance_opt:.6f}
    • Uniform Allocation Variance: {power_variance_uni:.6f}
    • Power Non-uniformity Factor: {power_variance_opt/power_variance_uni:.2f}x
    
    Efficiency Metrics:
    • Average Power Efficiency: {np.mean(opt_efficiency)/1e6:.2f} Mbps/W
    • Best Node Efficiency: {np.max(opt_efficiency)/1e6:.2f} Mbps/W
    • Worst Node Efficiency: {np.min(opt_efficiency)/1e6:.2f} Mbps/W
    
    Algorithm Conclusion:
    • Under current parameter settings, uniform
      power allocation is near-optimal
    • Channel conditions are relatively uniform,
      waterfilling effect is not significant
    • Consider testing more extreme channel
      conditions for better demonstration
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    filename = 'power_optimization_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Power optimization analysis chart saved: {filename}")
    
    # 显示图表
    plt.show()
    
    return results

def test_extreme_scenario():
    """
    测试极端信道条件下的注水算法效果
    """
    print("\n" + "="*60)
    print("🧪 Testing Extreme Channel Conditions Scenario")
    print("="*60)
    
    system = IoTSystem()
    optimizer = PowerOptimizer(system)
    
    # 创建极端的信道条件：一些节点很近，一些很远
    iot_positions = np.array([
        [349, 349, 0],    # 节点1: 极近
        [351, 351, 0],    # 节点2: 极近  
        [300, 300, 0],    # 节点3: 较近
        [500, 500, 0],    # 节点4: 较远
        [700, 700, 0],    # 节点5: 很远
        [900, 900, 0],    # 节点6: 极远
        [1200, 1200, 0],  # 节点7: 超远
        [348, 352, 0],    # 节点8: 极近
        [600, 600, 0],    # 节点9: 远
        [1000, 1000, 0]   # 节点10: 很远
    ])
    uav_position = np.array([350, 350, 100])
    
    # 执行优化
    results = optimizer.optimize_power_allocation(iot_positions, uav_position)
    
    # 详细分析
    optimizer.analyze_power_distribution(results)
    
    return results

if __name__ == "__main__":
    # 标准场景分析
    print("📊 Starting standard scenario power optimization analysis...")
    standard_results = create_power_analysis_plots()
    
    # 极端场景测试
    extreme_results = test_extreme_scenario()
