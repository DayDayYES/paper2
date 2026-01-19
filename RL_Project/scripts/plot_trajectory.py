"""
无人机飞行轨迹可视化脚本
从CSV日志中提取最优Episode的UAV轨迹并绘制2D/3D图
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.lines import Line2D

# 设置matplotlib参数，符合学术论文风格
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.width'] = 1.0
matplotlib.rcParams['ytick.major.width'] = 1.0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv_data(csv_path):
    """加载CSV日志数据"""
    print(f"尝试加载文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    # 使用encoding参数处理可能的编码问题
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"UTF-8编码读取失败，尝试其他编码: {e}")
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except Exception as e2:
            print(f"GBK编码也失败: {e2}")
            # 最后尝试用open打开再读取
            with open(csv_path, 'r', encoding='utf-8') as f:
                df = pd.read_csv(f)
    
    print(f"✓ 加载数据: {len(df)} 行")
    print(f"  Episode范围: {df['episode'].min()} ~ {df['episode'].max()}")
    print(f"  UAV数量: {df['uav_id'].nunique()}")
    return df


def find_best_episode(df):
    """找到reward最高的episode"""
    # 按episode分组，计算总reward
    episode_rewards = df.groupby('episode')['step_reward'].sum()
    best_episode = episode_rewards.idxmax()
    best_reward = episode_rewards.max()
    
    print(f"✓ 最优Episode: {best_episode} (Total Reward: {best_reward:.2f})")
    return best_episode


def extract_trajectory(df, episode):
    """提取指定episode的轨迹数据"""
    episode_data = df[df['episode'] == episode].copy()
    
    # 获取UAV数量
    uav_ids = sorted(episode_data['uav_id'].unique())
    
    trajectories = {}
    for uav_id in uav_ids:
        uav_data = episode_data[episode_data['uav_id'] == uav_id].sort_values('step')
        
        trajectories[uav_id] = {
            'steps': uav_data['step'].values,
            'x': uav_data['uav_x'].values,
            'y': uav_data['uav_y'].values,
            'z': uav_data['uav_z'].values,
            'user_ids': uav_data['user_id'].values,
            'ue_x': uav_data['ue_x'].values,
            'ue_y': uav_data['ue_y'].values
        }
    
    return trajectories, uav_ids


def get_user_positions(df, episode):
    """获取所有用户位置和服务状态"""
    episode_data = df[df['episode'] == episode]
    
    # 获取所有被服务的用户
    served_users = episode_data[episode_data['user_id'] >= 0][['user_id', 'ue_x', 'ue_y']].drop_duplicates()
    
    user_positions = {}
    for _, row in served_users.iterrows():
        user_id = int(row['user_id'])
        if user_id >= 0:
            user_positions[user_id] = (row['ue_x'], row['ue_y'])
    
    return user_positions


def plot_trajectory_2d(trajectories, uav_ids, user_positions, episode, save_dir):
    """绘制2D俯视图轨迹"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 颜色方案
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    
    # 绘制用户位置
    for user_id, (ux, uy) in user_positions.items():
        ax.scatter(ux, uy, c='gray', marker='o', s=80, alpha=0.6, zorder=1)
        ax.annotate(f'U{user_id}', (ux, uy), textcoords="offset points", 
                   xytext=(5, 5), fontsize=8, color='gray')
    
    # 绘制每个UAV的轨迹
    legend_elements = []
    for idx, uav_id in enumerate(uav_ids):
        traj = trajectories[uav_id]
        color = colors[idx % len(colors)]
        
        x, y = traj['x'], traj['y']
        user_ids = traj['user_ids']
        ue_x, ue_y = traj['ue_x'], traj['ue_y']
        
        # 绘制轨迹线
        ax.plot(x, y, color=color, linewidth=2.0, alpha=0.8, zorder=2)
        
        # 绘制轨迹点
        ax.scatter(x, y, c=color, s=50, zorder=3, alpha=0.8)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], c=color, marker='s', s=150, 
                  edgecolors='black', linewidths=2, zorder=5, label=f'UAV{uav_id} Start')
        ax.scatter(x[-1], y[-1], c=color, marker='^', s=150, 
                  edgecolors='black', linewidths=2, zorder=5)
        
        # 绘制服务连线（UAV到用户）
        for i in range(len(x)):
            if user_ids[i] >= 0 and ue_x[i] > 0:
                ax.plot([x[i], ue_x[i]], [y[i], ue_y[i]], 
                       color=color, linestyle='--', linewidth=1.0, alpha=0.4, zorder=1)
        
        # 绘制飞行方向箭头
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            if np.sqrt(dx**2 + dy**2) > 1:  # 只在有明显移动时画箭头
                ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                           zorder=4)
        
        # 图例元素
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                      marker='s', markersize=8, label=f'UAV {uav_id}'))
    
    # 添加用户图例
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='gray', markersize=8, label='Users'))
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'UAV Trajectory (2D) - Episode {episode}', fontsize=16, fontweight='bold')
    
    # 设置范围
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
    
    # 网格和图例
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存
    pdf_path = os.path.join(save_dir, f'trajectory_2d_episode_{episode}.pdf')
    png_path = os.path.join(save_dir, f'trajectory_2d_episode_{episode}.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 2D轨迹图已保存: {png_path}")
    return png_path


def plot_trajectory_3d(trajectories, uav_ids, user_positions, episode, save_dir):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色方案
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    
    # 绘制用户位置（在地面z=0）
    for user_id, (ux, uy) in user_positions.items():
        ax.scatter(ux, uy, 0, c='gray', marker='o', s=60, alpha=0.6)
    
    # 绘制每个UAV的轨迹
    legend_elements = []
    for idx, uav_id in enumerate(uav_ids):
        traj = trajectories[uav_id]
        color = colors[idx % len(colors)]
        
        x, y, z = traj['x'], traj['y'], traj['z']
        user_ids = traj['user_ids']
        ue_x, ue_y = traj['ue_x'], traj['ue_y']
        
        # 绘制3D轨迹线
        ax.plot(x, y, z, color=color, linewidth=2.5, alpha=0.9, zorder=3)
        
        # 绘制轨迹点
        ax.scatter(x, y, z, c=color, s=40, zorder=4, alpha=0.8)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], c=color, marker='s', s=150, 
                  edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], c=color, marker='^', s=150, 
                  edgecolors='black', linewidths=2, zorder=5)
        
        # 绘制到地面的投影线（虚线）
        for i in range(0, len(x), 2):  # 每隔一个点画一条
            ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], 0], 
                   color=color, linestyle=':', linewidth=0.8, alpha=0.3)
        
        # 绘制服务连线（UAV到用户）
        for i in range(len(x)):
            if user_ids[i] >= 0 and ue_x[i] > 0:
                ax.plot([x[i], ue_x[i]], [y[i], ue_y[i]], [z[i], 0],
                       color=color, linestyle='--', linewidth=1.0, alpha=0.4)
        
        # 图例元素
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2.5, 
                                      marker='s', markersize=8, label=f'UAV {uav_id}'))
    
    # 添加用户图例
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='gray', markersize=8, label='Users (z=0)'))
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'UAV Trajectory (3D) - Episode {episode}', fontsize=16, fontweight='bold', pad=20)
    
    # 设置范围
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
    ax.set_zlim(0, 600)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 图例
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 保存
    pdf_path = os.path.join(save_dir, f'trajectory_3d_episode_{episode}.pdf')
    png_path = os.path.join(save_dir, f'trajectory_3d_episode_{episode}.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 3D轨迹图已保存: {png_path}")
    return png_path


def main():
    """主函数"""
    # 路径设置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, 'results', 'logs', 'training_details.csv')
    save_dir = os.path.join(project_root, 'results', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 50)
    print("无人机飞行轨迹可视化")
    print("=" * 50)
    
    # 1. 加载数据
    df = load_csv_data(csv_path)
    
    # 2. 找到最优episode
    best_episode = find_best_episode(df)
    
    # 3. 提取轨迹数据
    trajectories, uav_ids = extract_trajectory(df, best_episode)
    print(f"✓ 提取轨迹: {len(uav_ids)} 个UAV")
    
    # 4. 获取用户位置
    user_positions = get_user_positions(df, best_episode)
    print(f"✓ 用户位置: {len(user_positions)} 个用户")
    
    # 5. 绘制2D轨迹图
    plot_trajectory_2d(trajectories, uav_ids, user_positions, best_episode, save_dir)
    
    # 6. 绘制3D轨迹图
    plot_trajectory_3d(trajectories, uav_ids, user_positions, best_episode, save_dir)
    
    print("=" * 50)
    print("轨迹图生成完成！")
    print(f"保存目录: {save_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()