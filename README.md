# SCA (连续凸近似) 算法实现

## 简介

这是一个完整的**连续凸近似 (Successive Convex Approximation, SCA)** 算法实现，用于求解非凸优化问题。该实现包含了算法核心代码、多个经典测试函数示例，以及丰富的可视化功能。

## 算法原理

SCA算法是一种迭代优化算法,通过在每个迭代点处构造原目标函数的凸近似,求解凸子问题来逐步逼近最优解。

**主要步骤:**
1. 在当前点 x_k 处构造目标函数的凸近似
2. 求解凸优化子问题得到新的候选点 x_{k+1}
3. 根据实际减少量和预测减少量更新信赖域半径
4. 检查收敛条件,重复迭代直到收敛

## 文件结构

```
.
├── sca_algorithm.py      # SCA算法核心实现
├── examples.py           # 示例程序
├── requirements.txt      # 依赖包列表
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速开始

运行示例程序:

```bash
python examples.py
```

程序会提示您选择要运行的示例:
- 示例1: Rosenbrock函数 (香蕉形山谷)
- 示例2: Rastrigin函数 (多峰函数)
- 示例3: Beale函数
- 示例4: 二次函数 (简单示例)
- 选项5: 运行所有示例

### 自定义使用

```python
import numpy as np
from sca_algorithm import SCAOptimizer, visualize_2d_optimization

# 定义目标函数
def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

# 定义梯度函数
def gradient(x):
    return np.array([2*(x[0] - 2), 2*(x[1] - 3)])

# 创建优化器
optimizer = SCAOptimizer(
    objective_func=objective,
    grad_func=gradient,
    x0=np.array([0.0, 0.0]),  # 初始点
    bounds=[(-5, 10), (-5, 10)],  # 变量边界
    max_iter=50,  # 最大迭代次数
    tol=1e-6,  # 收敛容差
    trust_region_radius=1.0,  # 初始信赖域半径
    shrink_factor=0.9  # 信赖域收缩因子
)

# 运行优化
x_opt, history = optimizer.optimize()

# 可视化结果
visualize_2d_optimization(optimizer, x_range=(-5, 10), y_range=(-5, 10))
```

## 主要功能

### SCAOptimizer 类

**参数:**
- `objective_func`: 目标函数 f(x)
- `grad_func`: 梯度函数 ∇f(x)
- `x0`: 初始点
- `bounds`: 变量边界 (可选)
- `max_iter`: 最大迭代次数 (默认: 50)
- `tol`: 收敛容差 (默认: 1e-6)
- `trust_region_radius`: 初始信赖域半径 (默认: 1.0)
- `shrink_factor`: 信赖域收缩因子 (默认: 0.9)

**方法:**
- `optimize()`: 执行优化,返回最优解和优化历史

### 可视化函数

1. **visualize_2d_optimization()**: 2D优化过程可视化
   - 等高线图 + 优化路径
   - 目标函数收敛曲线
   - 梯度范数变化

2. **visualize_3d_surface()**: 3D曲面可视化
   - 3D曲面 + 优化路径
   - 俯视图

3. **visualize_convergence_metrics()**: 收敛指标可视化
   - 目标函数值变化
   - 梯度范数变化
   - 信赖域半径变化
   - 相邻迭代差值

## 测试函数

### 1. Rosenbrock函数
```
f(x, y) = (1-x)² + 100(y-x²)²
全局最优解: (1, 1)
全局最优值: 0
```

### 2. Rastrigin函数
```
f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))
全局最优解: (0, 0)
全局最优值: 0
```

### 3. Beale函数
```
f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
全局最优解: (3, 0.5)
全局最优值: 0
```

### 4. 二次函数
```
f(x, y) = (x-2)² + (y-3)² + xy
```

## 输出结果

程序会生成以下可视化图片:
- `sca_optimization_2d.png`: 2D优化路径和收敛曲线
- `sca_optimization_3d.png`: 3D曲面和优化路径
- `sca_convergence_metrics.png`: 详细的收敛指标

## 算法特点

✅ **鲁棒性强**: 采用信赖域方法,保证算法稳定性
✅ **收敛快速**: 对于凸和某些非凸问题收敛速度快
✅ **可视化完善**: 提供多种可视化方式,便于理解优化过程
✅ **易于扩展**: 代码结构清晰,易于添加约束和修改

## 注意事项

- 对于强非凸问题,可能收敛到局部最优解
- 初始点和信赖域半径的选择会影响收敛速度和结果
- 建议根据具体问题调整参数

## 参考文献

- Beck, A., & Teboulle, M. (2012). "Smoothing and first order methods: A unified framework." SIAM Journal on Optimization.
- Marks, B. R., & Wright, G. P. (1978). "A general inner approximation algorithm for nonconvex mathematical programs." Operations Research.

## 许可证

MIT License

## 作者

AI Assistant

## 更新日志

- v1.0.0 (2025-10-17): 初始版本发布

