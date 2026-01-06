"""
连续凸近似 (Successive Convex Approximation, SCA) 算法实现
用于求解非凸优化问题
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果文件夹
if not os.path.exists('result'):
    os.makedirs('result')


class SCAOptimizer:
    """连续凸近似优化器"""
    
    def __init__(self, objective_func, grad_func, constraints=None, 
                 x0=None, bounds=None, max_iter=50, tol=1e-6, 
                 trust_region_radius=1.0, shrink_factor=0.9):
        """
        初始化SCA优化器
        
        参数:
            objective_func: 目标函数 f(x)
            grad_func: 梯度函数 grad_f(x)
            constraints: 约束函数列表 (可选)
            x0: 初始点
            bounds: 变量边界 [(min1, max1), (min2, max2), ...]
            max_iter: 最大迭代次数
            tol: 收敛容差
            trust_region_radius: 初始信赖域半径
            shrink_factor: 信赖域收缩因子
        """
        self.objective_func = objective_func
        self.grad_func = grad_func
        self.constraints = constraints or []
        self.x0 = x0
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.trust_region_radius = trust_region_radius
        self.shrink_factor = shrink_factor
        
        # 记录优化过程
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'trust_radius': []
        }
        
    def _convex_approximation(self, x_k, x):
        """
        在点x_k处构造目标函数的凸近似
        使用一阶泰勒展开 + 二次正则项
        """
        f_k = self.objective_func(x_k)
        grad_k = self.grad_func(x_k)
        
        # 凸近似: f(x_k) + grad_f(x_k)^T(x - x_k) + (1/(2*rho))||x - x_k||^2
        rho = self.trust_region_radius
        approx = f_k + np.dot(grad_k, x - x_k) + (1.0 / (2.0 * rho)) * np.sum((x - x_k)**2)
        
        return approx
    
    def _solve_convex_subproblem(self, x_k, line_search=True):
        """
        求解凸子问题
        min convex_approximation(x_k, x)
        s.t. ||x - x_k|| <= trust_region_radius
             bounds constraints
        
        使用线搜索的梯度法求解
        """
        grad_k = self.grad_func(x_k)
        f_k = self.objective_func(x_k)
        
        if line_search:
            # Armijo线搜索
            alpha = self.trust_region_radius
            beta = 0.5
            c1 = 1e-4
            
            for _ in range(20):  # 最多20次线搜索
                x_new = x_k - alpha * grad_k
                
                # 应用边界约束
                if self.bounds is not None:
                    for i, (lb, ub) in enumerate(self.bounds):
                        x_new[i] = np.clip(x_new[i], lb, ub)
                
                f_new = self.objective_func(x_new)
                
                # Armijo条件
                if f_new <= f_k - c1 * alpha * np.dot(grad_k, grad_k):
                    return x_new
                
                alpha *= beta
            
            # 如果线搜索失败,返回小步长
            alpha = 1e-6
        else:
            alpha = self.trust_region_radius
        
        # 计算步长
        x_new = x_k - alpha * grad_k
        
        # 应用边界约束
        if self.bounds is not None:
            for i, (lb, ub) in enumerate(self.bounds):
                x_new[i] = np.clip(x_new[i], lb, ub)
        
        return x_new
    
    def _update_trust_region(self, x_k, x_new, f_k, f_new):
        """更新信赖域半径"""
        # 计算实际减少量
        actual_reduction = f_k - f_new
        
        # 计算预测减少量
        predicted_reduction = (self._convex_approximation(x_k, x_k) - 
                              self._convex_approximation(x_k, x_new))
        
        # 计算增益比
        if abs(predicted_reduction) < 1e-10:
            rho = 0 if actual_reduction <= 0 else 1
        else:
            rho = actual_reduction / predicted_reduction
        
        # 更新信赖域半径
        if rho < 0.25:
            # 近似不好,缩小信赖域
            self.trust_region_radius *= self.shrink_factor
        elif rho > 0.75:
            # 近似很好,扩大信赖域
            self.trust_region_radius = min(self.trust_region_radius / self.shrink_factor, 1.0)
        
        # 接受条件: 实际减少量为正
        return actual_reduction > -1e-10  # 允许微小的增加
    
    def optimize(self):
        """执行SCA优化"""
        if self.x0 is None:
            raise ValueError("需要提供初始点 x0")
        
        x_k = np.array(self.x0, dtype=float)
        
        print("=" * 60)
        print("开始 SCA 优化")
        print("=" * 60)
        print(f"初始点: x0 = {x_k}")
        print(f"初始目标值: f(x0) = {self.objective_func(x_k):.6f}")
        print(f"最大迭代次数: {self.max_iter}")
        print(f"收敛容差: {self.tol}")
        print("=" * 60)
        
        for iteration in range(self.max_iter):
            # 计算当前状态
            f_k = self.objective_func(x_k)
            grad_k = self.grad_func(x_k)
            grad_norm = np.linalg.norm(grad_k)
            
            # 输出当前迭代信息
            print(f"迭代 {iteration + 1:3d}: f(x) = {f_k:10.6f}, "
                  f"||grad|| = {grad_norm:10.6e}, "
                  f"trust_radius = {self.trust_region_radius:.6f}")
            
            # 记录当前状态
            self.history['x'].append(x_k.copy())
            self.history['f'].append(f_k)
            self.history['grad_norm'].append(grad_norm)
            self.history['trust_radius'].append(self.trust_region_radius)
            
            # 检查收敛 - 梯度范数
            if grad_norm < self.tol:
                print("\n" + "=" * 60)
                print(f"✓ 算法收敛！梯度范数 {grad_norm:.6e} < {self.tol}")
                print("=" * 60)
                break
            
            # 求解凸子问题(使用线搜索)
            x_new = self._solve_convex_subproblem(x_k, line_search=True)
            
            # 计算新点的函数值
            f_new = self.objective_func(x_new)
            
            # 更新信赖域并决定是否接受
            accept = self._update_trust_region(x_k, x_new, f_k, f_new)
            
            if accept:
                # 检查步长和函数值变化
                step_norm = np.linalg.norm(x_new - x_k)
                f_change = abs(f_k - f_new)
                
                x_k = x_new
                
                # 多重收敛准则
                if step_norm < self.tol * 0.1:  # 步长极小
                    print("\n" + "=" * 60)
                    print(f"✓ 算法收敛！步长 {step_norm:.6e} 极小")
                    print("=" * 60)
                    break
                
                if f_change < self.tol * 0.1 and iteration > 10:  # 函数值变化极小
                    print("\n" + "=" * 60)
                    print(f"✓ 算法收敛！函数值变化 {f_change:.6e} 极小")
                    print("=" * 60)
                    break
            else:
                # 即使不接受,也可能因为信赖域太小而终止
                if self.trust_region_radius < 1e-10:
                    print("\n" + "=" * 60)
                    print(f"⚠ 信赖域半径过小 {self.trust_region_radius:.6e}，终止迭代")
                    print("=" * 60)
                    break
        
        # 最终结果
        if iteration == self.max_iter - 1:
            print("\n" + "=" * 60)
            print("! 达到最大迭代次数")
            print("=" * 60)
        
        print(f"\n最优解: x* = {x_k}")
        print(f"最优值: f(x*) = {self.objective_func(x_k):.6f}")
        print(f"总迭代次数: {len(self.history['f'])}")
        
        return x_k, self.history


def visualize_2d_optimization(optimizer, x_range, y_range, resolution=100, filename_prefix=''):
    """
    可视化2D优化过程
    
    参数:
        optimizer: 已运行的SCAOptimizer对象
        x_range: x轴范围 (min, max)
        y_range: y轴范围 (min, max)
        resolution: 网格分辨率
        filename_prefix: 文件名前缀
    """
    history = optimizer.history
    
    if len(history['x']) == 0:
        print("错误: 优化器尚未运行")
        return
    
    if len(history['x'][0]) != 2:
        print("错误: 仅支持2D可视化")
        return
    
    # 创建网格
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # 计算目标函数值
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = optimizer.objective_func(np.array([X[i, j], Y[i, j]]))
    
    # 创建图形
    fig = plt.figure(figsize=(18, 5))
    
    # 子图1: 等高线图 + 优化路径
    ax1 = fig.add_subplot(131)
    contour = ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # 绘制优化路径
    x_path = np.array([x[0] for x in history['x']])
    y_path = np.array([x[1] for x in history['x']])
    
    ax1.plot(x_path, y_path, 'r.-', linewidth=2, markersize=8, label='优化路径')
    ax1.plot(x_path[0], y_path[0], 'go', markersize=12, label='起点')
    ax1.plot(x_path[-1], y_path[-1], 'r*', markersize=15, label='终点')
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_title('优化路径 (等高线图)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 目标函数值变化
    ax2 = fig.add_subplot(132)
    iterations = range(1, len(history['f']) + 1)
    ax2.plot(iterations, history['f'], 'b-o', linewidth=2, markersize=6)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('目标函数值 f(x)', fontsize=12)
    ax2.set_title('目标函数收敛曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 梯度范数变化
    ax3 = fig.add_subplot(133)
    ax3.semilogy(iterations, history['grad_norm'], 'g-s', linewidth=2, markersize=6)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('梯度范数 ||∇f(x)||', fontsize=12)
    ax3.set_title('梯度范数变化 (对数坐标)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'result/{filename_prefix}sca_optimization_2d.png' if filename_prefix else 'result/sca_optimization_2d.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 2D可视化图已保存为 '{filename}'")
    plt.close()


def visualize_3d_surface(optimizer, x_range, y_range, resolution=50, filename_prefix=''):
    """
    3D曲面可视化
    
    参数:
        optimizer: 已运行的SCAOptimizer对象
        x_range: x轴范围 (min, max)
        y_range: y轴范围 (min, max)
        resolution: 网格分辨率
        filename_prefix: 文件名前缀
    """
    history = optimizer.history
    
    if len(history['x'][0]) != 2:
        return
    
    # 创建网格
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # 计算目标函数值
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = optimizer.objective_func(np.array([X[i, j], Y[i, j]]))
    
    # 创建3D图形
    fig = plt.figure(figsize=(14, 6))
    
    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, 
                           edgecolor='none', antialiased=True)
    
    # 绘制优化路径
    x_path = np.array([x[0] for x in history['x']])
    y_path = np.array([x[1] for x in history['x']])
    z_path = np.array(history['f'])
    
    ax1.plot(x_path, y_path, z_path, 'r.-', linewidth=3, markersize=8, label='优化路径')
    ax1.scatter(x_path[0], y_path[0], z_path[0], c='green', s=100, marker='o', label='起点')
    ax1.scatter(x_path[-1], y_path[-1], z_path[-1], c='red', s=150, marker='*', label='终点')
    
    ax1.set_xlabel('x₁', fontsize=12)
    ax1.set_ylabel('x₂', fontsize=12)
    ax1.set_zlabel('f(x)', fontsize=12)
    ax1.set_title('3D曲面 + 优化路径', fontsize=14, fontweight='bold')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 俯视图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.contour3D(X, Y, Z, 50, cmap='viridis', alpha=0.6)
    ax2.plot(x_path, y_path, z_path, 'r.-', linewidth=3, markersize=8)
    ax2.scatter(x_path[0], y_path[0], z_path[0], c='green', s=100, marker='o')
    ax2.scatter(x_path[-1], y_path[-1], z_path[-1], c='red', s=150, marker='*')
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlabel('x₁', fontsize=12)
    ax2.set_ylabel('x₂', fontsize=12)
    ax2.set_zlabel('f(x)', fontsize=12)
    ax2.set_title('俯视图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filename = f'result/{filename_prefix}sca_optimization_3d.png' if filename_prefix else 'result/sca_optimization_3d.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 3D可视化图已保存为 '{filename}'")
    plt.close()


def visualize_convergence_metrics(optimizer, filename_prefix=''):
    """
    可视化收敛指标
    
    参数:
        optimizer: 已运行的SCAOptimizer对象
        filename_prefix: 文件名前缀
    """
    history = optimizer.history
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iterations = range(1, len(history['f']) + 1)
    
    # 目标函数值
    ax1 = axes[0, 0]
    ax1.plot(iterations, history['f'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('目标函数值 f(x)', fontsize=12)
    ax1.set_title('目标函数值变化', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 梯度范数 (对数)
    ax2 = axes[0, 1]
    ax2.semilogy(iterations, history['grad_norm'], 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('梯度范数 ||∇f(x)||', fontsize=12)
    ax2.set_title('梯度范数变化 (对数坐标)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 信赖域半径
    ax3 = axes[1, 0]
    ax3.plot(iterations, history['trust_radius'], 'r-^', linewidth=2, markersize=6)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('信赖域半径', fontsize=12)
    ax3.set_title('信赖域半径变化', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 相邻迭代目标函数差值
    ax4 = axes[1, 1]
    f_diff = [abs(history['f'][i] - history['f'][i-1]) for i in range(1, len(history['f']))]
    ax4.semilogy(range(2, len(history['f']) + 1), f_diff, 'm-d', linewidth=2, markersize=6)
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('|f(xₖ) - f(xₖ₋₁)|', fontsize=12)
    ax4.set_title('相邻迭代目标函数差值 (对数坐标)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'result/{filename_prefix}sca_convergence_metrics.png' if filename_prefix else 'result/sca_convergence_metrics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 收敛指标图已保存为 '{filename}'")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SCA (连续凸近似) 算法演示")
    print("=" * 60)

