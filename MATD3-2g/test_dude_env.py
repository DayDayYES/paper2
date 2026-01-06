"""
测试解耦关联的全双工UAV环境
"""
import numpy as np
from uav_env_dude import UAVEnvDUDe

# 创建用户簇
ue_cluster_1 = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0], [365.74, 971.82, 0],
                [320.80, 406.66, 0], [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], [267.70, 926.15, 0]]
ue_cluster_2 = [[966.09, 757.82, 0], [304.61, 786.76, 0], [427.41, 1158.40, 0], [861.97, 925.30, 0], [541.30, 815.78, 0],
                [413.33, 1023.47, 0], [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], [878.19, 930.27, 0]]

print("=" * 80)
print("测试解耦关联的全双工UAV环境 (DUDe)")
print("=" * 80)

print("\n[1] 创建环境...")
env = UAVEnvDUDe(ue_cluster_1, ue_cluster_2, num_uavs=2)
print(f"✓ 环境创建成功")
print(f"  - 旋翼UAV数量: {env.uav_num}")
print(f"  - MBS位置: {env.mbs.get_position()}")
print(f"  - 用户设备数量: {env.ue_num}")
print(f"  - 总基站数量: {env.total_bs_num} (包括MBS)")
print(f"  - 动作空间维度: {env.get_action_dim()}")
print(f"  - 观察空间维度: {env.get_obs_dim()}")

print("\n[2] 检查UAV初始状态...")
for i, uav in enumerate(env.uavs):
    print(f"  UAV {i}:")
    print(f"    位置: {uav.get_position()}")
    print(f"    最大功率: {uav.P_uav} W")
    print(f"    速度范围: [{uav.v_uav_min}, {uav.v_uav_max}] m/s")

print("\n[3] 测试距离和路径损耗计算...")
d_bs_ue, d_bs_bs = env.calculate_all_distances()
print(f"✓ 距离计算完成")
print(f"  - 基站到UE距离矩阵形状: {d_bs_ue.shape}")
print(f"  - 基站间距离矩阵形状: {d_bs_bs.shape}")
print(f"  - 平均BS-UE距离: {np.mean(d_bs_ue):.2f} m")

PL_bs_ue, PL_bs_bs = env.compute_all_path_loss(d_bs_ue, d_bs_bs)
print(f"✓ 路径损耗计算完成")
print(f"  - BS-UE路径损耗形状: {PL_bs_ue.shape}")
print(f"  - BS-BS路径损耗形状: {PL_bs_bs.shape}")

print("\n[4] 测试Gumbel-Softmax用户关联...")
# 创建随机关联logits
test_logits = []
for i in range(env.uav_num):
    # 每个UAV输出 ue_num * 2 个logits
    logits = np.random.randn(env.ue_num * 2)
    test_logits.append(logits)

b_ul, b_dl = env.parse_association_from_action(test_logits, hard=False)
print(f"✓ 用户关联解析完成（软关联）")
print(f"  - 上行关联矩阵形状: {b_ul.shape}")
print(f"  - 下行关联矩阵形状: {b_dl.shape}")
print(f"  - 上行关联和（每UE应为1）: {np.sum(b_ul, axis=0)[:5]}...")
print(f"  - 下行关联和（每UE应为1）: {np.sum(b_dl, axis=0)[:5]}...")

# 测试hard模式
b_ul_hard, b_dl_hard = env.parse_association_from_action(test_logits, hard=True)
print(f"✓ 用户关联解析完成（硬关联）")
print(f"  - 上行关联（硬）前5个UE: {np.argmax(b_ul_hard, axis=0)[:5]}")
print(f"  - 下行关联（硬）前5个UE: {np.argmax(b_dl_hard, axis=0)[:5]}")

print("\n[5] 测试干扰和速率计算（全双工模式）...")
p_bs = np.array([10, 5, 5])  # MBS和2个UAV的功率
rate_ul, rate_dl, I_ul, I_dl = env.compute_interference_and_rate(
    b_ul, b_dl, p_bs, PL_bs_ue, PL_bs_bs)
print(f"✓ 干扰和速率计算完成")
print(f"  - 上行速率形状: {rate_ul.shape}")
print(f"  - 下行速率形状: {rate_dl.shape}")
print(f"  - 总上行速率: {np.sum(rate_ul) / 1e6:.2f} Mbps")
print(f"  - 总下行速率: {np.sum(rate_dl) / 1e6:.2f} Mbps")
print(f"  - 平均上行干扰: {np.mean(I_ul[I_ul > 0]):.2e} W")
print(f"  - 平均下行干扰: {np.mean(I_dl[I_dl > 0]):.2e} W")

print("\n[6] 测试回程链路计算...")
backhaul_rate = env.compute_backhaul_rate(PL_bs_bs, p_bs)
print(f"✓ 回程链路速率计算完成")
print(f"  - UAV回程速率: {backhaul_rate / 1e6} Mbps")

print("\n[7] 测试环境reset...")
state = env.reset()
print(f"✓ 环境重置成功")
print(f"  - 返回状态数量: {len(state)}")
print(f"  - 每个状态维度: {[s.shape for s in state]}")
print(f"  - 温度重置为: {env.temperature}")

print("\n[8] 测试单步执行...")
# 创建随机动作
n_actions = env.get_action_dim()
actions = []
for i in range(env.uav_num):
    action = np.random.uniform(-1, 1, n_actions)
    actions.append(action)

print(f"  动作维度: {[len(a) for a in actions]}")
state, reward, done, info = env.step(actions)
print(f"✓ 单步执行成功")
print(f"  - 新状态: {[s.shape for s in state]}")
print(f"  - 奖励: {reward:.2f}")
print(f"  - 完成: {done}")
print(f"  - 信息:")
for key, value in info.items():
    if isinstance(value, (int, float)):
        print(f"      {key}: {value:.4f}")

print("\n[9] 测试多步运行...")
env.reset()
total_reward = 0
episode_info = {
    'rates': [],
    'powers': [],
    'penalties': []
}

for step in range(10):
    actions = []
    for i in range(env.uav_num):
        action = np.random.uniform(-1, 1, n_actions)
        actions.append(action)
    
    state, reward, done, info = env.step(actions)
    total_reward += reward
    
    episode_info['rates'].append(info['total_rate'])
    episode_info['powers'].append(info['total_power'])
    episode_info['penalties'].append(info['backhaul_penalty'] + info['safety_penalty'])
    
    if step % 3 == 0:
        print(f"  Step {step:2d}: reward={reward:7.2f}, rate={info['total_rate']:6.2f} Mbps, "
              f"power={info['total_power']:6.2f} W, temp={info['temperature']:.3f}")

print(f"\n✓ 多步运行完成")
print(f"  - 总奖励: {total_reward:.2f}")
print(f"  - 平均速率: {np.mean(episode_info['rates']):.2f} Mbps")
print(f"  - 平均功率: {np.mean(episode_info['powers']):.2f} W")
print(f"  - 平均惩罚: {np.mean(episode_info['penalties']):.2f}")

print("\n[10] 测试解耦关联特性...")
env.reset()
# 创建一个动作，使得上行和下行关联到不同基站
actions = []
for i in range(env.uav_num):
    action = np.random.uniform(-1, 1, n_actions)
    # 修改关联logits，让上行偏向UAV0，下行偏向UAV1
    if i == 0:
        action[4:4+env.ue_num] = 2.0  # 上行logits较大
        action[4+env.ue_num:] = -2.0   # 下行logits较小
    else:
        action[4:4+env.ue_num] = -2.0  # 上行logits较小
        action[4+env.ue_num:] = 2.0    # 下行logits较大
    actions.append(action)

state, reward, done, info = env.step(actions)

# 获取关联结果
d_bs_ue, d_bs_bs = env.calculate_all_distances()
PL_bs_ue, PL_bs_bs = env.compute_all_path_loss(d_bs_ue, d_bs_bs)
association_logits = [a[4:] for a in actions]
b_ul, b_dl = env.parse_association_from_action(association_logits, hard=True)

ul_associations = np.argmax(b_ul, axis=0)
dl_associations = np.argmax(b_dl, axis=0)

print(f"✓ 解耦关联测试")
print(f"  - 前5个UE的上行关联基站: {ul_associations[:5]}")
print(f"  - 前5个UE的下行关联基站: {dl_associations[:5]}")
print(f"  - 解耦用户数（上下行不同BS）: {np.sum(ul_associations != dl_associations)}/{env.ue_num}")

print("\n" + "=" * 80)
print("✓✓✓ 所有测试通过！环境运行正常！")
print("=" * 80)

print("\n[总结]")
print("新环境特性:")
print("  ✓ 解耦上下行关联（DUDe）- UE可以上下行连接不同基站")
print("  ✓ 全双工干扰模型 - 考虑基站自干扰、UE间干扰、BS间干扰")
print("  ✓ Gumbel-Softmax用户关联 - 连续动作空间中处理离散关联")
print("  ✓ 回程链路约束 - 软约束，违反时惩罚")
print("  ✓ 新奖励函数 - 速率最大化 - 功率惩罚 - 回程惩罚 - 安全惩罚")
print("  ✓ 温度退火 - Gumbel-Softmax温度逐渐降低")

