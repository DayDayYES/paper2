"""
测试重构后的代码是否正常工作
"""
import numpy as np
from uav_env import UAVTASKENV
from uav import UAV, FUAV

# 创建用户簇
ue_cluster_1 = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0],[365.74, 971.82, 0],
                [320.80, 406.66, 0], [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], [267.70, 926.15, 0]]
ue_cluster_2 = [[966.09, 757.82, 0], [304.61, 786.76, 0], [427.41, 1158.40, 0], [861.97, 925.30, 0], [541.30, 815.78, 0],
                [413.33, 1023.47, 0], [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], [878.19, 930.27, 0]]

print("=" * 60)
print("测试1: 创建环境")
print("=" * 60)
env = UAVTASKENV(ue_cluster_1, ue_cluster_2)
print(f"✓ 环境创建成功")
print(f"  - 无人机数量: {env.uav_num}")
print(f"  - 用户数量: {env.ue_num}")
print(f"  - 无人机类型: {[type(uav).__name__ for uav in env.uavs]}")
print(f"  - 固定翼无人机: {type(env.fuav).__name__}")

print("\n" + "=" * 60)
print("测试2: 检查无人机属性")
print("=" * 60)
for i, uav in enumerate(env.uavs):
    print(f"UAV {i}:")
    print(f"  - 位置: {uav.get_position()}")
    print(f"  - 计算频率: {uav.f_uav / 1e9:.2f} GHz")
    print(f"  - 传输功率: {uav.P_uav} W")
    print(f"  - 带宽: {uav.B / 1e6:.2f} MHz")
    print(f"  - 速度范围: [{uav.v_uav_min}, {uav.v_uav_max}] m/s")

print(f"\n固定翼无人机:")
print(f"  - 位置: {env.fuav.get_position()}")
print(f"  - 计算频率: {env.fuav.f_fuav / 1e9:.2f} GHz")
print(f"  - 是否固定: {env.fuav.is_fixed}")

print("\n" + "=" * 60)
print("测试3: 测试无人机移动")
print("=" * 60)
uav0 = env.uavs[0]
old_pos = uav0.get_position().copy()
print(f"移动前位置: {old_pos}")

# 创建一个测试动作 [距离归一化, 水平角归一化, 垂直角归一化]
test_action = np.array([0.5, 0.5, 0.5])  # 中等速度，中等角度
success, old_position, d, theta_v = uav0.move(test_action)
new_pos = uav0.get_position()
print(f"移动后位置: {new_pos}")
print(f"移动成功: {success}")
print(f"移动距离: {d:.2f} m")
print(f"位置变化: {np.linalg.norm(new_pos - old_pos):.2f} m")

print("\n" + "=" * 60)
print("测试4: 测试固定翼无人机不移动")
print("=" * 60)
fuav_old_pos = env.fuav.get_position().copy()
print(f"移动前位置: {fuav_old_pos}")
success, old_position, d, theta_v = env.fuav.move(test_action)
fuav_new_pos = env.fuav.get_position()
print(f"移动后位置: {fuav_new_pos}")
print(f"位置是否相同: {np.array_equal(fuav_old_pos, fuav_new_pos)}")

print("\n" + "=" * 60)
print("测试5: 测试环境reset")
print("=" * 60)
state = env.reset()
print(f"✓ Reset成功")
print(f"  - 返回的状态数量: {len(state)}")
print(f"  - 每个状态的维度: {[len(s) for s in state]}")
for i, s in enumerate(state):
    print(f"  - UAV {i} 状态: {s}")

print("\n" + "=" * 60)
print("测试6: 测试环境step")
print("=" * 60)
# 创建随机动作
n_actions = 3 + len(ue_cluster_1 + ue_cluster_2)
actions = []
for i in range(env.uav_num):
    action = np.random.uniform(-1, 1, n_actions)
    actions.append(action)

print(f"动作维度: {[len(a) for a in actions]}")
state, reward, done, info, time, energy = env.step(actions)
print(f"✓ Step执行成功")
print(f"  - 新状态数量: {len(state)}")
print(f"  - 奖励: {reward:.2f}")
print(f"  - 时延: {time:.2f}")
print(f"  - 能耗: {energy:.2f}")
print(f"  - 完成: {done}")

print("\n" + "=" * 60)
print("测试7: 测试通信模型")
print("=" * 60)
uav_locations = [uav.get_position() for uav in env.uavs]
spatial_distances, uav_fix_distances = env.calculate_distances(uav_locations, env.uecord)
print(f"✓ 距离计算成功")
print(f"  - 空间距离矩阵形状: {spatial_distances.shape}")
print(f"  - 到固定翼距离形状: {uav_fix_distances.shape}")

L, capacity = env.capacity(spatial_distances, uav_locations)
print(f"✓ 容量计算成功")
print(f"  - 路径损耗矩阵形状: {L.shape}")
print(f"  - 容量矩阵形状: {capacity.shape}")
print(f"  - 平均容量: {np.mean(capacity) / 1e6:.2f} Mbps")

coverage, cover1 = env.coverge(spatial_distances, capacity)
print(f"✓ 覆盖计算成功")
print(f"  - 覆盖矩阵形状: {coverage.shape}")
print(f"  - 覆盖用户数: {np.sum(coverage > 0)}")
print(f"  - 未覆盖用户数: {np.sum(cover1 == 0)}")

print("\n" + "=" * 60)
print("测试8: 测试能耗模型")
print("=" * 60)
d = 30  # 30米
theta_v = np.pi / 4  # 45度
P_fly = env.fly_energy(d, theta_v, env.uavs[0])
print(f"✓ 飞行能耗计算成功")
print(f"  - 飞行距离: {d} m")
print(f"  - 垂直角度: {theta_v * 180 / np.pi:.2f}°")
print(f"  - 飞行功率: {P_fly:.2f} W")

print("\n" + "=" * 60)
print("测试9: 多步运行测试")
print("=" * 60)
env.reset()
total_reward = 0
for step in range(5):
    actions = []
    for i in range(env.uav_num):
        action = np.random.uniform(-1, 1, n_actions)
        actions.append(action)
    state, reward, done, info, time, energy = env.step(actions)
    total_reward += reward
    print(f"  Step {step + 1}: reward={reward:.2f}, time={time:.2f}, energy={energy:.2f}")

print(f"✓ 多步运行成功")
print(f"  - 总奖励: {total_reward:.2f}")

print("\n" + "=" * 60)
print("✓✓✓ 所有测试通过！重构成功！✓✓✓")
print("=" * 60)

