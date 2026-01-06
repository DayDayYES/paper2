"""
SCA算法示例 - 求解不同类型的优化问题
"""

import numpy as np
from sca_algorithm import SCAOptimizer, visualize_2d_optimization, visualize_3d_surface, visualize_convergence_metrics


def example_1_rosenbrock():
    """
    示例1: Rosenbrock函数 (经典非凸优化问题)
    f(x, y) = (1-x)^2 + 100(y-x^2)^2
    全局最优解: (1, 1), 最优值: 0
    """
    print("\n" + "=" * 60)
    print("示例 1: Rosenbrock函数")
    print("=" * 60)
    print("函数: f(x, y) = (1-x)² + 100(y-x²)²")
    print("全局最优解: (1, 1)")
    print("全局最优值: 0")
    print("=" * 60)
    
    def objective(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def gradient(x):
        dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dy = 200 * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    # 初始化优化器
    x0 = np.array([-1.0, 1.0])
    optimizer = SCAOptimizer(
        objective_func=objective,
        grad_func=gradient,
        x0=x0,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=200,
        tol=1e-5,
        trust_region_radius=0.1,
        shrink_factor=0.8
    )
    
    # 运行优化
    x_opt, history = optimizer.optimize()
    
    # 可视化
    print("\n开始生成可视化图...")
    visualize_2d_optimization(optimizer, x_range=(-2, 2), y_range=(-1, 3), filename_prefix='rosenbrock_')
    visualize_3d_surface(optimizer, x_range=(-2, 2), y_range=(-1, 3), filename_prefix='rosenbrock_')
    visualize_convergence_metrics(optimizer, filename_prefix='rosenbrock_')
    
    return optimizer


def example_2_rastrigin():
    """
    示例2: Rastrigin函数 (多峰非凸函数)
    f(x, y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))
    全局最优解: (0, 0), 最优值: 0
    
    注意: 这是一个高度多峰函数,SCA算法容易陷入局部最优
    """
    print("\n" + "=" * 60)
    print("示例 2: Rastrigin函数 (多峰函数)")
    print("=" * 60)
    print("函数: f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))")
    print("全局最优解: (0, 0)")
    print("全局最优值: 0")
    print("⚠ 注意: 此函数有大量局部最优,初始点很重要!")
    print("=" * 60)
    
    def objective(x):
        return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    
    def gradient(x):
        dx = 2*x[0] + 20*np.pi*np.sin(2*np.pi*x[0])
        dy = 2*x[1] + 20*np.pi*np.sin(2*np.pi*x[1])
        return np.array([dx, dy])
    
    # 初始化优化器  
    # 注意: Rastrigin是高度多峰函数,从(2,2)出发可能陷入局部最优
    # 建议使用接近原点的初始点
    x0 = np.array([0.5, 0.5])  # 更接近全局最优的初始点
    optimizer = SCAOptimizer(
        objective_func=objective,
        grad_func=gradient,
        x0=x0,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=200,
        tol=1e-5,
        trust_region_radius=0.1,
        shrink_factor=0.8
    )
    
    # 运行优化
    x_opt, history = optimizer.optimize()
    
    # 可视化
    print("\n开始生成可视化图...")
    visualize_2d_optimization(optimizer, x_range=(-5, 5), y_range=(-5, 5), filename_prefix='rastrigin_')
    visualize_3d_surface(optimizer, x_range=(-5, 5), y_range=(-5, 5), filename_prefix='rastrigin_')
    visualize_convergence_metrics(optimizer, filename_prefix='rastrigin_')
    
    return optimizer


def example_3_beale():
    """
    示例3: Beale函数
    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    全局最优解: (3, 0.5), 最优值: 0
    """
    print("\n" + "=" * 60)
    print("示例 3: Beale函数")
    print("=" * 60)
    print("函数: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²")
    print("全局最优解: (3, 0.5)")
    print("全局最优值: 0")
    print("=" * 60)
    
    def objective(x):
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return term1 + term2 + term3
    
    def gradient(x):
        term1 = 1.5 - x[0] + x[0]*x[1]
        term2 = 2.25 - x[0] + x[0]*x[1]**2
        term3 = 2.625 - x[0] + x[0]*x[1]**3
        
        dx = 2*term1*(-1 + x[1]) + 2*term2*(-1 + x[1]**2) + 2*term3*(-1 + x[1]**3)
        dy = 2*term1*x[0] + 2*term2*2*x[0]*x[1] + 2*term3*3*x[0]*x[1]**2
        
        return np.array([dx, dy])
    
    # 初始化优化器
    x0 = np.array([1.0, 1.0])  # 更接近最优解的初始点
    optimizer = SCAOptimizer(
        objective_func=objective,
        grad_func=gradient,
        x0=x0,
        bounds=[(-4.5, 4.5), (-4.5, 4.5)],
        max_iter=200,
        tol=1e-5,
        trust_region_radius=0.1,
        shrink_factor=0.8
    )
    
    # 运行优化
    x_opt, history = optimizer.optimize()
    
    # 可视化
    print("\n开始生成可视化图...")
    visualize_2d_optimization(optimizer, x_range=(-1, 4), y_range=(-1, 2), filename_prefix='beale_')
    visualize_3d_surface(optimizer, x_range=(-1, 4), y_range=(-1, 2), filename_prefix='beale_')
    visualize_convergence_metrics(optimizer, filename_prefix='beale_')
    
    return optimizer


def example_4_quadratic():
    """
    示例4: 简单二次函数
    f(x, y) = (x-2)^2 + (y-3)^2 + xy
    """
    print("\n" + "=" * 60)
    print("示例 4: 二次函数")
    print("=" * 60)
    print("函数: f(x, y) = (x-2)² + (y-3)² + xy")
    print("=" * 60)
    
    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 3)**2 + x[0]*x[1]
    
    def gradient(x):
        dx = 2*(x[0] - 2) + x[1]
        dy = 2*(x[1] - 3) + x[0]
        return np.array([dx, dy])
    
    # 初始化优化器
    x0 = np.array([5.0, 5.0])
    optimizer = SCAOptimizer(
        objective_func=objective,
        grad_func=gradient,
        x0=x0,
        bounds=[(-5, 10), (-5, 10)],
        max_iter=200,
        tol=1e-5,
        trust_region_radius=0.2,
        shrink_factor=0.8
    )
    
    # 运行优化
    x_opt, history = optimizer.optimize()
    
    # 可视化
    print("\n开始生成可视化图...")
    visualize_2d_optimization(optimizer, x_range=(-2, 8), y_range=(-2, 8), filename_prefix='quadratic_')
    visualize_3d_surface(optimizer, x_range=(-2, 8), y_range=(-2, 8), filename_prefix='quadratic_')
    visualize_convergence_metrics(optimizer, filename_prefix='quadratic_')
    
    return optimizer


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 15 + "SCA算法示例程序" + " " * 27 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)
    
    print("\n请选择要运行的示例:")
    print("1. Rosenbrock函数 (香蕉形山谷)")
    print("2. Rastrigin函数 (多峰函数)")
    print("3. Beale函数")
    print("4. 二次函数 (简单示例)")
    print("5. 运行所有示例")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    examples = {
        '1': example_1_rosenbrock,
        '2': example_2_rastrigin,
        '3': example_3_beale,
        '4': example_4_quadratic
    }
    
    if choice in examples:
        examples[choice]()
    elif choice == '5':
        for key in ['1', '2', '3', '4']:
            examples[key]()
            print("\n" + "=" * 60)
            print("按Enter继续下一个示例...")
            print("=" * 60)
            input()
    else:
        print("无效选择，默认运行示例1...")
        example_1_rosenbrock()
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + " " * 18 + "程序运行完成!" + " " * 23 + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)

