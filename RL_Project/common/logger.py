"""
训练日志记录器与专业绘图模块
符合强化学习论文作图风格
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from scipy.ndimage import uniform_filter1d

# 设置matplotlib参数，符合学术论文风格
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['xtick.major.width'] = 1.0
matplotlib.rcParams['ytick.major.width'] = 1.0
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.framealpha'] = 0.8
matplotlib.rcParams['legend.edgecolor'] = 'gray'

from config.config import LogConfig


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_file):
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        
        # 构建日志文件路径
        if not os.path.isabs(log_file):
            log_dir = os.path.join(self.project_root, 'results', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, log_file)
        else:
            self.log_file = log_file
        
        # 创建图片保存目录
        self.figures_dir = os.path.join(self.project_root, 'results', 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.episode_data = []
        
        # 清空日志文件
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
    
    def log(self, message, print_console=True):
        """记录日志"""
        if print_console:
            print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def log_episode(self, episode, data):
        """记录每回合数据"""
        self.episode_data.append(data)
        
        if LogConfig.SAVE_DETAILED_LOG:
            message = (f"\n{'='*60}\n"
                      f"Episode {episode}\n"
                      f"  Avg Score: {data['avg_score']:.2f}\n"
                      f"  Total Reward: {data['total_reward']:.2f}\n")
            
            if 'total_delay' in data:
                message += (f"  Total Delay: {data['total_delay']:.4f} s\n"
                           f"  Avg Delay: {data['avg_delay']:.4f} s\n"
                           f"  Total Energy: {data['total_energy']:.2f} J\n"
                           f"  Avg Energy: {data['avg_energy']:.2f} J\n"
                           f"  Scheduled: {data['scheduled_count']}\n"
                           f"  Steps: {data['steps']}\n")
            
            self.log(message, print_console=False)
    
    def _smooth_data(self, data, window_size=10):
        """
        平滑数据（使用滑动平均）
        
        Args:
            data: 原始数据
            window_size: 窗口大小
        
        Returns:
            smoothed: 平滑后的数据
        """
        if len(data) < window_size:
            return data
        return uniform_filter1d(data, size=window_size, mode='nearest')
    
    def _compute_confidence_interval(self, data, window_size=50):
        """
        计算置信区间（用于绘制阴影区域）
        
        Args:
            data: 原始数据
            window_size: 窗口大小
        
        Returns:
            mean: 均值
            lower: 下界
            upper: 上界
        """
        n = len(data)
        if n < window_size:
            return data, data, data
        
        mean = []
        lower = []
        upper = []
        
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2)
            window = data[start:end]
            
            m = np.mean(window)
            std = np.std(window)
            
            mean.append(m)
            lower.append(m - std)
            upper.append(m + std)
        
        return np.array(mean), np.array(lower), np.array(upper)
    
    def plot_results(self):
        """绘制训练结果（符合RL论文风格）"""
        if len(self.episode_data) == 0:
            return
        
        # 提取数据
        episodes = np.array([d['episode'] for d in self.episode_data])
        rewards = np.array([d['total_reward'] for d in self.episode_data])
        avg_scores = np.array([d['avg_score'] for d in self.episode_data])
        
        # 1. 绘制奖励曲线（主图）
        self._plot_reward_curve(episodes, rewards, avg_scores)
        
        # 2. 如果有时延和能耗数据，绘制额外图表
        if 'total_delay' in self.episode_data[0]:
            delays = np.array([d['total_delay'] for d in self.episode_data])
            energies = np.array([d['total_energy'] for d in self.episode_data])
            
            # 绘制时延曲线
            self._plot_delay_curve(episodes, delays)
            
            # 绘制能耗曲线
            self._plot_energy_curve(episodes, energies)
            
            # 绘制综合对比图
            self._plot_combined_metrics(episodes, rewards, delays, energies)
    
    def _plot_reward_curve(self, episodes, rewards, avg_scores):
        """
        绘制奖励曲线（RL论文标准风格）
        - 原始曲线用浅色
        - 平滑曲线用深色
        - 添加阴影表示方差
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算平滑曲线和置信区间
        smooth_rewards = self._smooth_data(rewards, window_size=20)
        mean, lower, upper = self._compute_confidence_interval(rewards, window_size=50)
        
        # 绘制阴影区域（置信区间）
        ax.fill_between(episodes, lower, upper, alpha=0.2, color='#2196F3', label='Std Dev')
        
        # 绘制原始曲线（浅色）
        ax.plot(episodes, rewards, alpha=0.3, color='#2196F3', linewidth=0.8)
        
        # 绘制平滑曲线（深色）
        ax.plot(episodes, smooth_rewards, color='#1565C0', linewidth=2.0, label='MATD3 (Smoothed)')
        
        # 设置坐标轴
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
        ax.set_title('Training Performance', fontsize=16, fontweight='bold', pad=10)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 设置图例
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        
        # 设置刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.figures_dir, 'reward_curve.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, 'reward_curve.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_delay_curve(self, episodes, delays):
        """绘制时延曲线"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算平滑曲线和置信区间
        smooth_delays = self._smooth_data(delays, window_size=20)
        mean, lower, upper = self._compute_confidence_interval(delays, window_size=50)
        
        # 绘制阴影区域
        ax.fill_between(episodes, lower, upper, alpha=0.2, color='#4CAF50')
        
        # 绘制原始曲线
        ax.plot(episodes, delays, alpha=0.3, color='#4CAF50', linewidth=0.8)
        
        # 绘制平滑曲线
        ax.plot(episodes, smooth_delays, color='#2E7D32', linewidth=2.0, label='Task Delay (Smoothed)')
        
        # 设置坐标轴
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Delay (s)', fontsize=14, fontweight='bold')
        ax.set_title('Task Completion Delay', fontsize=16, fontweight='bold', pad=10)
        
        # 设置网格和图例
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, 'delay_curve.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, 'delay_curve.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_curve(self, episodes, energies):
        """绘制能耗曲线"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 计算平滑曲线和置信区间
        smooth_energies = self._smooth_data(energies, window_size=20)
        mean, lower, upper = self._compute_confidence_interval(energies, window_size=50)
        
        # 绘制阴影区域
        ax.fill_between(episodes, lower, upper, alpha=0.2, color='#FF9800')
        
        # 绘制原始曲线
        ax.plot(episodes, energies, alpha=0.3, color='#FF9800', linewidth=0.8)
        
        # 绘制平滑曲线
        ax.plot(episodes, smooth_energies, color='#E65100', linewidth=2.0, label='Energy Consumption (Smoothed)')
        
        # 设置坐标轴
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Energy (J)', fontsize=14, fontweight='bold')
        ax.set_title('Energy Consumption', fontsize=16, fontweight='bold', pad=10)
        
        # 设置网格和图例
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, 'energy_curve.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, 'energy_curve.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_combined_metrics(self, episodes, rewards, delays, energies):
        """
        绘制综合对比图（子图形式）
        符合论文多指标对比风格
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # 颜色方案
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        dark_colors = ['#1565C0', '#2E7D32', '#E65100']
        
        data_list = [rewards, delays, energies]
        labels = ['Episode Reward', 'Total Delay (s)', 'Total Energy (J)']
        titles = ['(a) Reward', '(b) Delay', '(c) Energy']
        
        for idx, (ax, data, label, title, color, dark_color) in enumerate(
            zip(axes, data_list, labels, titles, colors, dark_colors)):
            
            # 计算平滑曲线
            smooth_data = self._smooth_data(data, window_size=20)
            mean, lower, upper = self._compute_confidence_interval(data, window_size=50)
            
            # 绘制阴影区域
            ax.fill_between(episodes, lower, upper, alpha=0.2, color=color)
            
            # 绘制原始曲线
            ax.plot(episodes, data, alpha=0.3, color=color, linewidth=0.8)
            
            # 绘制平滑曲线
            ax.plot(episodes, smooth_data, color=dark_color, linewidth=2.0, label='MATD3')
            
            # 设置坐标轴
            ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            
            # 设置刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # 设置边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
            
            # 只在第一个子图显示图例
            if idx == 0:
                ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, 'combined_metrics.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, 'combined_metrics.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, other_data_dict, save_name='algorithm_comparison'):
        """
        绘制多算法对比图（用于论文中的算法对比）
        
        Args:
            other_data_dict: 其他算法的数据字典
                            {'Algorithm Name': {'episodes': [...], 'rewards': [...]}}
            save_name: 保存文件名
        """
        if len(self.episode_data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 颜色和标记方案
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
        markers = ['o', 's', '^', 'D', 'v']
        
        # 绘制当前算法（MATD3）
        episodes = np.array([d['episode'] for d in self.episode_data])
        rewards = np.array([d['total_reward'] for d in self.episode_data])
        smooth_rewards = self._smooth_data(rewards, window_size=20)
        
        ax.plot(episodes, smooth_rewards, color=colors[0], linewidth=2.0, 
                label='MATD3 (Proposed)', marker=markers[0], markevery=len(episodes)//10,
                markersize=6)
        
        # 绘制其他算法
        for idx, (name, data) in enumerate(other_data_dict.items()):
            color_idx = (idx + 1) % len(colors)
            marker_idx = (idx + 1) % len(markers)
            
            other_episodes = np.array(data['episodes'])
            other_rewards = np.array(data['rewards'])
            smooth_other = self._smooth_data(other_rewards, window_size=20)
            
            ax.plot(other_episodes, smooth_other, color=colors[color_idx], 
                   linewidth=1.8, label=name, marker=markers[marker_idx],
                   markevery=len(other_episodes)//10, markersize=5)
        
        # 设置坐标轴
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
        ax.set_title('Algorithm Comparison', fontsize=16, fontweight='bold', pad=10)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 设置图例
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9, ncol=1)
        
        # 设置刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, f'{save_name}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, f'{save_name}.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis(self, window_sizes=[10, 50, 100]):
        """
        绘制收敛性分析图（展示不同平滑窗口的效果）
        """
        if len(self.episode_data) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        episodes = np.array([d['episode'] for d in self.episode_data])
        rewards = np.array([d['total_reward'] for d in self.episode_data])
        
        # 绘制原始曲线
        ax.plot(episodes, rewards, alpha=0.3, color='gray', linewidth=0.8, label='Raw')
        
        # 绘制不同窗口大小的平滑曲线
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        for idx, ws in enumerate(window_sizes):
            smooth = self._smooth_data(rewards, window_size=ws)
            ax.plot(episodes, smooth, color=colors[idx % len(colors)], 
                   linewidth=1.8, label=f'Window={ws}')
        
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
        ax.set_title('Convergence Analysis', fontsize=16, fontweight='bold', pad=10)
        
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, 'convergence_analysis.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        save_path_png = os.path.join(self.figures_dir, 'convergence_analysis.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
