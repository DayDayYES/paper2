"""
可视化UAV和UE在二维平面的位置分布
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import EnvironmentConfig
from environments.uav_env_dude import UAVEnvDUDe
from environments.ENV import Environment as LegacyEnvironment


def plot_uav_ue_layout(env, save_path=None, show_plot=True):
    """
    绘制UAV和UE的二维平面布局图
    
    Args:
        env: 环境实例
        save_path: 保存路径（可选）
        show_plot: 是否显示图形
    
    Returns:
        fig, ax: matplotlib图形和坐标轴对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 获取UE位置
    if hasattr(env, 'uecord'):
        ue_positions = env.uecord
    elif hasattr(env, 'cluster1') and hasattr(env, 'cluster2'):
        ue_positions = env.cluster1 + env.cluster2
    else:
        raise ValueError("无法获取UE位置信息")
    
    # 获取UAV位置
    uav_positions = []
    if hasattr(env, 'uavs'):
        for uav in env.uavs:
            pos = uav.get_position()
            uav_positions.append(pos)
    elif hasattr(env, 'state'):
        uav_positions = env.state
    
    # 获取MBS位置（如果有）
    mbs_position = None
    if hasattr(env, 'mbs'):
        mbs_position = env.mbs.get_position()
    
    # 提取x, y坐标（忽略z坐标）
    ue_x = [pos[0] for pos in ue_positions]
    ue_y = [pos[1] for pos in ue_positions]
    
    uav_x = [pos[0] for pos in uav_positions]
    uav_y = [pos[1] for pos in uav_positions]
    
    # 绘制UE（用户设备）
    ax.scatter(ue_x, ue_y, c='blue', s=150, marker='o', 
               edgecolors='darkblue', linewidths=2, 
               label='User Equipment (UE)', zorder=3, alpha=0.7)
    
    # 给UE标注索引
    for i, (x, y) in enumerate(zip(ue_x, ue_y)):
        ax.annotate(f'UE{i}', (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='blue', alpha=0.7),
                   zorder=4)
    
    # 绘制UAV（旋翼无人机）
    ax.scatter(uav_x, uav_y, c='red', s=300, marker='^', 
               edgecolors='darkred', linewidths=2,
               label='UAV', zorder=3, alpha=0.8)
    
    # 给UAV标注索引
    for i, (x, y) in enumerate(zip(uav_x, uav_y)):
        ax.annotate(f'UAV{i+1}', (x, y), xytext=(0, -20), 
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                            edgecolor='red', alpha=0.8),
                   zorder=4)
    
    # 绘制MBS（固定翼无人机/宏基站）
    if mbs_position is not None:
        mbs_x, mbs_y = mbs_position[0], mbs_position[1]
        ax.scatter(mbs_x, mbs_y, c='green', s=300, marker='s', 
                   edgecolors='darkgreen', linewidths=2,
                   label='MBS (Fixed-wing UAV)', zorder=3, alpha=0.8)
        ax.annotate('MBS', (mbs_x, mbs_y), xytext=(0, -25), 
                   textcoords='offset points', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                            edgecolor='green', alpha=0.8),
                   zorder=4)
    
    # 绘制用户簇区域（可选）
    if hasattr(env, 'cluster1') and hasattr(env, 'cluster2'):
        # 簇1的中心和范围
        cluster1_x = [pos[0] for pos in env.cluster1]
        cluster1_y = [pos[1] for pos in env.cluster1]
        if len(cluster1_x) > 0:
            center1_x, center1_y = np.mean(cluster1_x), np.mean(cluster1_y)
            radius1 = max([np.sqrt((x-center1_x)**2 + (y-center1_y)**2) 
                          for x, y in zip(cluster1_x, cluster1_y)]) + 50
            circle1 = Circle((center1_x, center1_y), radius1, 
                           fill=False, linestyle='--', 
                           color='blue', linewidth=1.5, alpha=0.5)
            ax.add_patch(circle1)
        
        # 簇2的中心和范围
        cluster2_x = [pos[0] for pos in env.cluster2]
        cluster2_y = [pos[1] for pos in env.cluster2]
        if len(cluster2_x) > 0:
            center2_x, center2_y = np.mean(cluster2_x), np.mean(cluster2_y)
            radius2 = max([np.sqrt((x-center2_x)**2 + (y-center2_y)**2) 
                          for x, y in zip(cluster2_x, cluster2_y)]) + 50
            circle2 = Circle((center2_x, center2_y), radius2, 
                           fill=False, linestyle='--', 
                           color='purple', linewidth=1.5, alpha=0.5)
            ax.add_patch(circle2)
    
    # 设置坐标轴
    ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_title('UAV and UE Layout in 2D Plane', fontsize=14, fontweight='bold')
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    
    # 设置图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 设置坐标轴范围（添加一些边距）
    all_x = ue_x + uav_x
    all_y = ue_y + uav_y
    if mbs_position is not None:
        all_x.append(mbs_x)
        all_y.append(mbs_y)
    
    x_margin = (max(all_x) - min(all_x)) * 0.1
    y_margin = (max(all_y) - min(all_y)) * 0.1
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    
    # 添加统计信息文本框
    # info_text = (f'Total UAVs: {len(uav_positions)}\n'
    #             f'Total UEs: {len(ue_positions)}\n'
    #             f'Area: {max(all_x)-min(all_x):.0f} × {max(all_y)-min(all_y):.0f} m²')
    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
    #        fontsize=10, verticalalignment='top',
    #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图形已保存到: {save_path}")
    
    # 显示图形
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def main():
    """主函数"""
    print("="*70)
    print("UAV和UE位置可视化")
    print("="*70)
    
    # 创建环境
    env_type = EnvironmentConfig.ENV_TYPE
    print(f"\n创建 {env_type} 环境...")
    
    if env_type == 'dude':
        env = UAVEnvDUDe(
            EnvironmentConfig.UE_CLUSTER_1,
            EnvironmentConfig.UE_CLUSTER_2,
            num_uavs=EnvironmentConfig.NUM_UAVS
        )
    else:
        env = LegacyEnvironment(
            EnvironmentConfig.UE_CLUSTER_1,
            EnvironmentConfig.UE_CLUSTER_2
        )
    
    print(f"✓ 环境创建成功")
    print(f"  - UAV数量: {env.uav_num}")
    print(f"  - UE数量: {len(EnvironmentConfig.UE_CLUSTER_1 + EnvironmentConfig.UE_CLUSTER_2)}")
    
    # 生成保存路径
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'uav_ue_layout.png')
    
    # 绘制图形
    print(f"\n绘制布局图...")
    plot_uav_ue_layout(env, save_path=save_path, show_plot=True)
    
    print("\n✓ 完成！")


if __name__ == '__main__':
    main()