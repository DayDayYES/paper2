"""
快速开始脚本 - 快速测试新环境和训练流程
运行少量回合以验证代码正确性
"""
import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EnvironmentConfig, TrainingConfig
from uav_env_dude import UAVEnvDUDe
from perddpg_torch import Agent


def quick_test():
    """快速测试环境和训练"""
    print("="*70)
    print(" "*20 + "快速开始测试")
    print("="*70)
    
    # 1. 创建环境
    print("\n[1/5] 创建环境...")
    env = UAVEnvDUDe(
        EnvironmentConfig.UE_CLUSTER_1,
        EnvironmentConfig.UE_CLUSTER_2,
        num_uavs=2
    )
    print(f"✓ 环境创建成功")
    print(f"  - UAV数量: {env.uav_num}")
    print(f"  - UE数量: {env.ue_num}")
    print(f"  - 观察维度: {env.get_obs_dim()}")
    print(f"  - 动作维度: {env.get_action_dim()}")
    
    # 2. 创建智能体
    print("\n[2/5] 创建智能体...")
    test_batch_size = 16  # 使用较小的批大小用于快速测试
    agents = []
    for i in range(env.uav_num):
        agent = Agent(
            alpha=TrainingConfig.ALPHA,
            beta=TrainingConfig.BETA,
            input_dims=env.get_obs_dim(),
            tau=TrainingConfig.TAU,
            n_actions=env.get_action_dim(),
            gamma=TrainingConfig.GAMMA,
            max_size=10000,  # 减小缓冲区用于测试
            C_fc1_dims=TrainingConfig.CRITIC_FC1_DIMS,
            C_fc2_dims=TrainingConfig.CRITIC_FC2_DIMS,
            C_fc3_dims=TrainingConfig.CRITIC_FC3_DIMS,
            A_fc1_dims=TrainingConfig.ACTOR_FC1_DIMS,
            A_fc2_dims=TrainingConfig.ACTOR_FC2_DIMS,
            batch_size=test_batch_size,  # 使用测试批大小
            n_agents=env.uav_num,
            noise=TrainingConfig.NOISE
        )
        agents.append(agent)
    print(f"✓ 创建 {len(agents)} 个智能体 (batch_size={test_batch_size})")
    
    # 3. 测试单步执行
    print("\n[3/5] 测试单步执行...")
    obs = env.reset()
    print(f"  - 初始状态: {[o.shape for o in obs]}")
    
    actions = []
    for i, agent in enumerate(agents):
        action = agent.choose_action(obs[i])
        action = np.clip(action, -1, 1)
        actions.append(action)
    print(f"  - 动作形状: {[a.shape for a in actions]}")
    
    obs_, reward, done, info = env.step(actions)
    print(f"  - 新状态: {[o.shape for o in obs_]}")
    print(f"  - 奖励: {reward:.2f}")
    print(f"  - 速率: {info['total_rate']:.2f} Mbps")
    print(f"  - 功率: {info['total_power']:.2f} W")
    print("✓ 单步执行成功")
    
    # 4. 测试训练循环
    print("\n[4/5] 测试训练循环 (5个回合)...")
    test_episodes = 5
    test_timestamp = 10
    total_steps = 0
    
    for episode in range(test_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_rate = 0
        
        for t in range(test_timestamp):
            # 选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.choose_action(obs[i])
                action = np.clip(action, -1, 1)
                actions.append(action)
            
            # 执行动作
            obs_, reward, done, info = env.step(actions)
            
            # 存储经验
            for i, agent in enumerate(agents):
                agent.remember((obs[i], actions[i], reward, obs_[i], done))
            
            total_steps += 1
            
            # 学习（缓冲区足够大时，并使用正确的批大小）
            if total_steps > test_batch_size * 2:  # 确保有足够的经验
                for agent in agents:
                    agent.learn(num_iteration=1, batch_size1=test_batch_size)
            
            obs = obs_
            episode_reward += reward
            episode_rate += info['total_rate']
        
        avg_rate = episode_rate / test_timestamp
        print(f"  Episode {episode+1}: Reward={episode_reward:7.2f}, "
              f"Avg Rate={avg_rate:6.2f} Mbps")
    
    print("✓ 训练循环测试成功")
    
    # 5. 测试环境特性
    print("\n[5/5] 测试环境特性...")
    
    # 测试Gumbel-Softmax关联
    test_logits = [np.random.randn(env.ue_num * 2) for _ in range(env.uav_num)]
    b_ul, b_dl = env.parse_association_from_action(test_logits, hard=True)
    ul_assoc = np.argmax(b_ul, axis=0)
    dl_assoc = np.argmax(b_dl, axis=0)
    
    print(f"  - 上行关联前5个UE: {ul_assoc[:5]}")
    print(f"  - 下行关联前5个UE: {dl_assoc[:5]}")
    print(f"  - 解耦用户数: {np.sum(ul_assoc != dl_assoc)}/{env.ue_num}")
    
    # 测试温度退火
    initial_temp = env.temperature
    for _ in range(100):
        env.step(actions)
    final_temp = env.temperature
    print(f"  - 温度退火: {initial_temp:.3f} → {final_temp:.3f}")
    
    print("✓ 环境特性测试成功")
    
    # 总结
    print("\n" + "="*70)
    print(" "*25 + "测试通过！")
    print("="*70)
    print("\n下一步:")
    print("  1. 运行完整训练: python train_dude.py")
    print("  2. 修改配置文件: config.py")
    print("  3. 查看训练指南: TRAINING_GUIDE.md")
    print("\n祝训练顺利！🚀")


if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

