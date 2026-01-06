# MATD3 训练指南 - 解耦关联的全双工UAV环境

## 📋 目录
1. [修改说明](#修改说明)
2. [文件结构](#文件结构)
3. [使用方法](#使用方法)
4. [配置说明](#配置说明)
5. [新旧环境对比](#新旧环境对比)

---

## 🔄 修改说明

### 新增文件

#### 1. `config.py` - 配置文件
**作用**: 集中管理所有训练超参数和环境配置

**主要内容**:
- `EnvironmentConfig`: 环境配置（用户簇、UAV数量、环境类型）
- `TrainingConfig`: 训练超参数（学习率、批大小、网络结构等）
- `LogConfig`: 日志和输出配置

**优势**:
- ✅ 参数集中管理，易于调整
- ✅ 避免硬编码，提高可维护性
- ✅ 支持多环境切换

#### 2. `train_dude.py` - 模块化训练脚本
**作用**: 新的规范化训练脚本，支持两种环境

**主要改进**:
1. **模块化设计**
   - `TrainingLogger`: 日志记录器类
   - `create_environment()`: 环境创建函数
   - `create_agents()`: 智能体创建函数
   - `train_episode()`: 单回合训练函数
   - `main()`: 主训练流程

2. **功能增强**
   - 支持进度条显示（tqdm）
   - 详细的日志记录
   - 自动绘制多种性能图表
   - 模型定期保存
   - 两种环境兼容

3. **代码规范**
   - 完整的文档字符串
   - 清晰的函数职责
   - 统一的命名规范

#### 3. `uav_env_dude.py` - 新环境实现
**作用**: 实现解耦关联的全双工UAV通信环境

**核心特性**:
- ✅ 解耦上下行关联（DUDe）
- ✅ 全双工干扰模型
- ✅ Gumbel-Softmax用户关联
- ✅ 回程链路约束
- ✅ 新奖励函数设计

---

## 📁 文件结构

```
MATD3-2g/
├── config.py                 # [新增] 配置文件
├── train_dude.py            # [新增] 模块化训练脚本
├── uav_env_dude.py          # [新增] 解耦关联环境
├── test_dude_env.py         # [新增] 环境测试脚本
│
├── train.py                 # [原有] 原始训练脚本
├── ENV.py                   # [原有] 原始环境（已重构）
├── uav.py                   # [原有] UAV实体类
│
├── perddpg_torch.py         # 算法实现
├── networks.py              # 网络结构
├── examplebuffer.py         # 经验回放缓冲区
└── ...
```

---

## 🚀 使用方法

### 方法1: 使用新训练脚本（推荐）

#### 步骤1: 配置参数
编辑 `config.py`:

```python
class EnvironmentConfig:
    NUM_UAVS = 2              # 设置UAV数量
    ENV_TYPE = 'dude'         # 选择环境类型: 'dude' 或 'legacy'

class TrainingConfig:
    N_EPISODES = 800          # 训练回合数
    TIMESTAMP = 100           # 每回合时间步
    ALPHA = 0.0001            # Actor学习率
    BETA = 0.001              # Critic学习率
    # ... 其他参数
```

#### 步骤2: 运行训练
```bash
python train_dude.py
```

#### 步骤3: 查看结果
训练完成后会生成：
- `training_log_dude.txt` - 详细训练日志
- `Avg_reward_dude.png` - 平均奖励曲线
- `Reward_dude.png` - 每回合奖励曲线
- `Rate_dude.png` - 数据速率曲线（仅DUDe环境）

---

### 方法2: 使用原训练脚本

如果想使用原始训练脚本，需要修改 `train.py`:

#### 修改1: 导入新环境
```python
# 原代码:
from uav_env import UAVTASKENV

# 修改为:
from uav_env_dude import UAVEnvDUDe
```

#### 修改2: 创建环境
```python
# 原代码:
env = UAVTASKENV(ue_cluster_1, ue_cluster_2)

# 修改为:
env = UAVEnvDUDe(ue_cluster_1, ue_cluster_2, num_uavs=2)
```

#### 修改3: 调整动作维度
```python
# 原代码:
n_actions = 3 + len(ue_cluster_1 + ue_cluster_2)

# 修改为:
n_actions = env.get_action_dim()  # DUDe环境: 4 + ue_num * 2
```

#### 修改4: 处理返回值
```python
# 原代码:
obs_, reward, done, info, t_, f_energy = env.step(action_all)

# 修改为:
obs_, reward, done, info = env.step(action_all)
# DUDe环境通过info字典返回额外信息
total_rate = info.get('total_rate', 0)
total_power = info.get('total_power', 0)
```

---

## ⚙️ 配置说明

### 环境配置 (EnvironmentConfig)

| 参数 | 说明 | 默认值 | 备注 |
|------|------|--------|------|
| `UE_CLUSTER_1` | 用户簇1坐标 | 10个UE | 可自定义 |
| `UE_CLUSTER_2` | 用户簇2坐标 | 10个UE | 可自定义 |
| `NUM_UAVS` | UAV数量 | 2 | 1-3个推荐 |
| `ENV_TYPE` | 环境类型 | 'dude' | 'dude' 或 'legacy' |

### 训练配置 (TrainingConfig)

| 参数 | 说明 | 默认值 | 调整建议 |
|------|------|--------|----------|
| `N_EPISODES` | 训练回合数 | 800 | 根据收敛情况调整 |
| `TIMESTAMP` | 每回合时间步 | 100 | 影响训练时间 |
| `BATCH_SIZE` | 批大小 | 64 | GPU性能允许可增大 |
| `ALPHA` | Actor学习率 | 0.0001 | 太大可能不稳定 |
| `BETA` | Critic学习率 | 0.001 | 通常大于ALPHA |
| `GAMMA` | 折扣因子 | 0.99 | 0.95-0.99之间 |
| `NOISE` | 探索噪声 | 0.2 | 训练后期可降低 |

### 网络结构配置

```python
# Critic网络 (3层)
CRITIC_FC1_DIMS = 512
CRITIC_FC2_DIMS = 256
CRITIC_FC3_DIMS = 256

# Actor网络 (2层)
ACTOR_FC1_DIMS = 1024
ACTOR_FC2_DIMS = 512
```

**调整建议**:
- 增大网络容量可以提高表达能力，但训练更慢
- 减小网络容量训练更快，但可能欠拟合
- 对于复杂环境（更多UAV/UE），建议增大网络

---

## 🔄 新旧环境对比

### 原始环境 (ENV.py / Legacy)

**特点**:
- 计算卸载场景
- 耦合上下行关联（同一基站）
- 简化的干扰模型
- 卸载比例作为动作

**动作空间** (维度: 3 + ue_num):
```
[速度, 水平角, 垂直角, offload_ratio_1, offload_ratio_2, ...]
```

**返回值**:
```python
state, reward, done, info, time, energy
```

**奖励函数**:
```python
reward = fairness - (time + energy) + penalties
```

---

### 新环境 (uav_env_dude.py / DUDe)

**特点**:
- ✅ 通信场景（上下行链路）
- ✅ 解耦关联（上下行可选不同基站）
- ✅ 完整全双工干扰模型
- ✅ Gumbel-Softmax处理离散关联
- ✅ 回程链路约束

**动作空间** (维度: 4 + ue_num × 2):
```
[速度, 水平角, 垂直角, UAV功率, 
 ue0_ul_logit, ue0_dl_logit,
 ue1_ul_logit, ue1_dl_logit,
 ...
 ue19_ul_logit, ue19_dl_logit]
```

**返回值**:
```python
state, reward, done, info
# info包含: total_rate, rate_ul, rate_dl, total_power, 
#           backhaul_penalty, safety_penalty, temperature
```

**奖励函数**:
```python
reward = w_rate × total_rate - 
         w_power × total_power - 
         w_backhaul × backhaul_penalty - 
         w_safety × safety_penalty
```

---

## 📊 性能指标

### Legacy环境指标
- Time (时延)
- Energy (能耗)
- Fairness (公平性)
- Coverage (覆盖率)

### DUDe环境指标
- **Total Rate**: 总数据速率 (Mbps)
- **UL Rate**: 上行速率 (Mbps)
- **DL Rate**: 下行速率 (Mbps)
- **Total Power**: 总功率 (W)
- **Backhaul Penalty**: 回程链路违反惩罚
- **Safety Penalty**: 安全距离违反惩罚
- **Temperature**: Gumbel-Softmax温度

---

## 🐛 调试技巧

### 1. 测试环境
运行测试脚本验证环境正常:
```bash
python test_dude_env.py
```

### 2. 查看日志
训练过程中实时查看日志:
```bash
tail -f training_log_dude.txt
```

### 3. 调整温度
如果关联不稳定，调整Gumbel-Softmax温度:
```python
# 在 uav_env_dude.py 中
self.temperature = 1.0      # 初始温度（更大=更平滑）
self.temp_min = 0.5         # 最小温度（更小=更离散）
self.temp_decay = 0.999     # 衰减率（更小=衰减更快）
```

### 4. 平衡奖励
调整奖励权重:
```python
# 在 uav_env_dude.py 中
self.w_rate = 1.0           # 速率权重（增大关注速率）
self.w_power = 0.01         # 功率权重（增大关注能耗）
self.w_backhaul = 0.1       # 回程惩罚权重
self.w_safety = 5.0         # 安全惩罚权重
```

---

## 📈 预期结果

### 收敛情况
- **前100 episodes**: 探索阶段，奖励波动大
- **100-400 episodes**: 学习阶段，奖励逐渐提升
- **400+ episodes**: 收敛阶段，奖励趋于稳定

### 典型指标（DUDe环境）
- **总速率**: 50-150 Mbps
- **功率**: 10-30 W
- **平均奖励**: 根据权重设置而定

---

## 💡 常见问题

### Q1: 训练很慢怎么办？
**A**: 
1. 减小 `N_EPISODES` 或 `TIMESTAMP`
2. 减小网络规模
3. 使用GPU（确保PyTorch支持CUDA）

### Q2: 奖励始终为负怎么办？
**A**: 
1. 调整奖励权重（降低惩罚权重）
2. 检查初始位置设置
3. 增加探索噪声

### Q3: 如何切换回原环境？
**A**: 在 `config.py` 中设置:
```python
ENV_TYPE = 'legacy'
```

### Q4: 如何保存和加载模型？
**A**: 
```python
# 保存（自动，每100回合）
# 或手动调用
for agent in agents:
    agent.save_models()

# 加载
for agent in agents:
    agent.load_models()
```

---

## 📝 总结

### 主要改进
1. ✅ **模块化设计**: 配置、环境、训练分离
2. ✅ **规范化代码**: 文档字符串、命名规范
3. ✅ **功能增强**: 日志、进度条、多图表
4. ✅ **灵活配置**: 易于切换环境和调参
5. ✅ **新环境支持**: 解耦关联、全双工、Gumbel-Softmax

### 下一步
1. 根据收敛情况调整超参数
2. 尝试不同的奖励权重配置
3. 对比两种环境的性能
4. 保存最优模型用于测试

---

**如有问题，请参考代码注释或联系开发者！**

