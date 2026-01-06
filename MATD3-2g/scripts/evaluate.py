"""
测试/评估已训练模型
加载保存的模型并在环境中测试性能
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入配置
from config import EnvironmentConfig, TrainingConfig

# 导入环境
from uav_env_dude import UAVEnvDUDe
from ENV import Environment as LegacyEnvironment

# 导入算法
from perddpg_torch import Agent


def create_environment(env_type):
    """创建环境"""
    if env_type == 'dude':
        env = UAVEnvDUDe(
            EnvironmentConfig.UE_CLUSTER_1,
            EnvironmentConfig.UE_CLUSTER_2,
            num_uavs=EnvironmentConfig.NUM_UAVS
        )
    elif env_type == 'legacy':
        env = LegacyEnvironment(
            EnvironmentConfig.UE_CLUSTER_1,
            EnvironmentConfig.UE_CLUSTER_2
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return env


def create_agents(env, config):
    """创建智能体（不训练，只用于加载模型）"""
    agents = []
    n_agents = env.uav_num
    
    # 获取观察和动作维度
    if hasattr(env, 'get_obs_dim'):
        obs_dim = env.get_obs_dim()
        action_dim = env.get_action_dim()
    else:
        obs_dim = 3
        action_dim = 3 + env.ue_num
    
    for i in range(n_agents):
        agent = Agent(
            alpha=config.ALPHA,
            beta=config.BETA,
            input_dims=obs_dim,
            tau=config.TAU,
            n_actions=action_dim,
            gamma=config.GAMMA,
            max_size=config.MEMORY_SIZE,
            C_fc1_dims=config.CRITIC_FC1_DIMS,
            C_fc2_dims=config.CRITIC_FC2_DIMS,
            C_fc3_dims=config.CRITIC_FC3_DIMS,
            A_fc1_dims=config.ACTOR_FC1_DIMS,
            A_fc2_dims=config.ACTOR_FC2_DIMS,
            batch_size=config.BATCH_SIZE,
            n_agents=n_agents,
            noise=0.0  # 测试时不添加噪声
        )
        agents.append(agent)
    
    return agents


def load_models(agents, model_dir='tmp/ddpg'):
    """
    加载所有智能体的模型
    
    Args:
        agents: 智能体列表
        model_dir: 模型保存目录
    
    Returns:
        bool: 是否成功加载
    """
    print(f"\n正在从 '{model_dir}' 加载模型...")
    
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        print(f"❌ 错误: 模型目录 '{model_dir}' 不存在！")
        return False
    
    # 检查模型文件是否存在
    required_files = ['actor_td3', 'target_actor_td3', 'critic_ddpg', 'target_critic_ddpg']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 错误: 缺少以下模型文件: {missing_files}")
        return False
    
    # 加载每个智能体的模型
    success_count = 0
    for i, agent in enumerate(agents):
        try:
            print(f"  加载智能体 {i+1} 的模型...")
            agent.load_models()
            success_count += 1
            print(f"  ✓ 智能体 {i+1} 模型加载成功")
        except Exception as e:
            print(f"  ❌ 智能体 {i+1} 模型加载失败: {e}")
    
    if success_count == len(agents):
        print(f"\n✓ 所有 {len(agents)} 个智能体的模型加载成功！")
        return True
    else:
        print(f"\n⚠ 警告: 只有 {success_count}/{len(agents)} 个智能体加载成功")
        return False


def evaluate_model(env, agents, num_episodes=10, num_steps=100):
    """
    评估模型性能
    
    Args:
        env: 环境实例
        agents: 智能体列表
        num_episodes: 测试回合数
        num_steps: 每回合步数
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n开始评估模型...")
    print(f"  测试回合数: {num_episodes}")
    print(f"  每回合步数: {num_steps}")
    
    episode_rewards = []
    episode_rates = []
    episode_powers = []
    episode_info_list = []
    
    for episode in tqdm(range(num_episodes), desc="评估进度"):
        obs = env.reset()
        episode_reward = 0
        episode_rate = 0
        episode_power = 0
        
        for step in range(num_steps):
            # 选择动作（不添加噪声）
            actions = []
            for i, agent in enumerate(agents):
                # 测试时不添加噪声
                action = agent.choose_action(obs[i])
                action = np.clip(action, -1, 1)
                actions.append(action)
            
            # 执行动作
            if hasattr(env, 'get_obs_dim'):  # DUDe环境
                obs_, reward, done, info = env.step(actions)
                episode_rate += info.get('total_rate', 0)
                episode_power += info.get('total_power', 0)
            else:  # Legacy环境
                obs_, reward, done, info, time_, energy_ = env.step(actions)
                episode_power += energy_
            
            obs = obs_
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_rates.append(episode_rate / num_steps if hasattr(env, 'get_obs_dim') else 0)
        episode_powers.append(episode_power / num_steps)
        
        episode_info = {
            'episode': episode + 1,
            'reward': episode_reward,
            'avg_rate': episode_rate / num_steps if hasattr(env, 'get_obs_dim') else 0,
            'avg_power': episode_power / num_steps
        }
        episode_info_list.append(episode_info)
    
    # 计算统计信息
    results = {
        'episode_rewards': episode_rewards,
        'episode_rates': episode_rates,
        'episode_powers': episode_powers,
        'episode_info': episode_info_list,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_rate': np.mean(episode_rates),
        'mean_power': np.mean(episode_powers),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }
    
    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*70)
    print("评估结果汇总")
    print("="*70)
    print(f"\n奖励统计:")
    print(f"  平均奖励: {results['mean_reward']:.2f}")
    print(f"  标准差:   {results['std_reward']:.2f}")
    print(f"  最小奖励: {results['min_reward']:.2f}")
    print(f"  最大奖励: {results['max_reward']:.2f}")
    
    if results['mean_rate'] > 0:
        print(f"\n数据速率统计:")
        print(f"  平均速率: {results['mean_rate']:.2f} Mbps")
    
    print(f"\n功率统计:")
    print(f"  平均功率: {results['mean_power']:.2f} W")
    
    print(f"\n各回合详情:")
    print(f"{'回合':<6} {'奖励':<12} {'速率(Mbps)':<15} {'功率(W)':<12}")
    print("-"*70)
    for info in results['episode_info']:
        rate_str = f"{info['avg_rate']:.2f}" if info['avg_rate'] > 0 else "N/A"
        print(f"{info['episode']:<6} {info['reward']:<12.2f} {rate_str:<15} {info['avg_power']:<12.2f}")
    
    print("="*70)


def plot_results(results, save_path='evaluation_results.png'):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 奖励曲线
    axes[0].plot(results['episode_rewards'], 'b-', linewidth=2, label='Episode Reward')
    axes[0].axhline(y=results['mean_reward'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0].fill_between(range(len(results['episode_rewards'])),
                        results['mean_reward'] - results['std_reward'],
                        results['mean_reward'] + results['std_reward'],
                        alpha=0.2, color='gray', label='±1 Std')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Model Evaluation - Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 速率曲线（如果有）
    if results['mean_rate'] > 0:
        axes[1].plot(results['episode_rates'], 'g-', linewidth=2, label='Data Rate')
        axes[1].axhline(y=results['mean_rate'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_rate"]:.2f} Mbps')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Data Rate (Mbps)')
        axes[1].set_title('Model Evaluation - Data Rates')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].plot(results['episode_powers'], 'orange', linewidth=2, label='Power')
        axes[1].axhline(y=results['mean_power'], color='r', linestyle='--',
                       label=f'Mean: {results["mean_power"]:.2f} W')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Power (W)')
        axes[1].set_title('Model Evaluation - Power Consumption')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ 评估结果图表已保存到: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("="*70)
    print("模型测试/评估脚本")
    print("="*70)
    
    # 配置
    model_dir = 'tmp/ddpg'  # 模型目录
    env_type = EnvironmentConfig.ENV_TYPE  # 环境类型
    num_episodes = 10  # 测试回合数
    num_steps = 100    # 每回合步数
    
    # 创建环境
    print(f"\n[1/4] 创建环境 ({env_type})...")
    env = create_environment(env_type)
    print(f"✓ 环境创建成功")
    
    # 创建智能体
    print(f"\n[2/4] 创建智能体...")
    agents = create_agents(env, TrainingConfig)
    print(f"✓ 创建 {len(agents)} 个智能体")
    
    # 加载模型
    print(f"\n[3/4] 加载模型...")
    if not load_models(agents, model_dir):
        print("\n❌ 模型加载失败，无法继续测试！")
        return
    
    # 评估模型
    print(f"\n[4/4] 评估模型性能...")
    results = evaluate_model(env, agents, num_episodes=num_episodes, num_steps=num_steps)
    
    # 打印结果
    print_results(results)
    
    # 绘制结果
    plot_results(results)
    
    print("\n✓ 评估完成！")


if __name__ == '__main__':
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 运行测试
    main()