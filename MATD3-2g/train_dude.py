"""
MATD3训练脚本 - 支持解耦关联的全双工UAV环境
模块化和规范化版本
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入配置
from config import EnvironmentConfig, TrainingConfig, LogConfig

# 导入环境
from uav_env_dude import UAVEnvDUDe
from ENV import Environment as LegacyEnvironment

# 导入算法
from perddpg_torch import Agent


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.episode_data = []
        
        # 清空日志文件
        if os.path.exists(log_file):
            os.remove(log_file)
    
    def log(self, message, print_console=True):
        """记录日志"""
        if print_console:
            print(message)
        with open('training_log.txt', 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def log_episode(self, episode, data):
        """记录每回合数据"""
        self.episode_data.append(data)
        
        if LogConfig.SAVE_DETAILED_LOG:
            message = (f"\n{'='*60}\n"
                      f"Episode {episode}\n"
                      f"  Avg Score: {data['avg_score']:.2f}\n"
                      f"  Total Reward: {data['total_reward']:.2f}\n")
            
            if 'total_rate' in data:
                message += (f"  Total Rate: {data['total_rate']:.2f} Mbps\n"
                           f"  UL Rate: {data['ul_rate']:.2f} Mbps\n"
                           f"  DL Rate: {data['dl_rate']:.2f} Mbps\n"
                           f"  Total Power: {data['total_power']:.2f} W\n")
            elif 'time' in data:
                message += (f"  Time: {data['time']:.2f}\n"
                           f"  Energy: {data['energy']:.2f}\n")
            
            self.log(message, print_console=False)
    
    def plot_results(self):
        """绘制训练结果"""
        if len(self.episode_data) == 0:
            return
        
        episodes = [d['episode'] for d in self.episode_data]
        avg_scores = [d['avg_score'] for d in self.episode_data]
        total_rewards = [d['total_reward'] for d in self.episode_data]
        
        # 绘制平均奖励
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, avg_scores, linewidth=2)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Training Performance - Average Reward', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(LogConfig.AVG_REWARD_PLOT, dpi=300)
        plt.close()
        
        # 绘制总奖励
        plt.figure(figsize=(10, 6))
        plt.plot(total_rewards, linewidth=1, alpha=0.6)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title('Training Performance - Episode Reward', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(LogConfig.REWARD_PLOT, dpi=300)
        plt.close()
        
        # 如果是DUDe环境，绘制速率图
        if 'total_rate' in self.episode_data[0]:
            total_rates = [d['total_rate'] for d in self.episode_data]
            ul_rates = [d['ul_rate'] for d in self.episode_data]
            dl_rates = [d['dl_rate'] for d in self.episode_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, total_rates, label='Total Rate', linewidth=2)
            plt.plot(episodes, ul_rates, label='Uplink Rate', linewidth=1, alpha=0.7)
            plt.plot(episodes, dl_rates, label='Downlink Rate', linewidth=1, alpha=0.7)
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Data Rate (Mbps)', fontsize=12)
            plt.title('Training Performance - Data Rate', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(LogConfig.RATE_PLOT, dpi=300)
            plt.close()


def create_environment(env_type):
    """
    创建环境
    
    Args:
        env_type: 环境类型 ('legacy' 或 'dude')
    
    Returns:
        env: 环境实例
    """
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
    """
    创建智能体
    
    Args:
        env: 环境实例
        config: 训练配置
    
    Returns:
        agents: 智能体列表
    """
    agents = []
    n_agents = env.uav_num
    
    # 获取观察和动作维度
    if hasattr(env, 'get_obs_dim'):
        obs_dim = env.get_obs_dim()
        action_dim = env.get_action_dim()
    else:
        obs_dim = 3  # legacy环境
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
            noise=config.NOISE,
            policy_delay=config.POLICY_DELAY
        )
        agents.append(agent)
    
    return agents


def train_episode(env, agents, timestamp, action_bound):
    """
    训练一个回合
    
    Args:
        env: 环境实例
        agents: 智能体列表
        timestamp: 时间步数
        action_bound: 动作边界
    
    Returns:
        episode_data: 回合数据字典
    """
    obs = env.reset()
    episode_reward = 0
    episode_info = {
        'rates': [],
        'powers': [],
        'ul_rates': [],
        'dl_rates': []
    }
    
    for t in range(timestamp):
        # 选择动作
        actions = []
        for i, agent in enumerate(agents):
            action = agent.choose_action(obs[i])
            action = np.clip(action, *action_bound)
            actions.append(action)
        
        # 执行动作
        if hasattr(env, 'get_obs_dim'):  # DUDe环境
            obs_, reward, done, info = env.step(actions)
            episode_info['rates'].append(info.get('total_rate', 0))
            episode_info['powers'].append(info.get('total_power', 0))
            episode_info['ul_rates'].append(info.get('rate_ul', 0))
            episode_info['dl_rates'].append(info.get('rate_dl', 0))
        else:  # Legacy环境
            obs_, reward, done, info, time_, energy_ = env.step(actions)
            episode_info['rates'].append(0)  # 占位
            episode_info['powers'].append(energy_)
        
        # 存储经验
        for i, agent in enumerate(agents):
            agent.remember((obs[i], actions[i], reward, obs_[i], done))
        
        # 学习
        for agent in agents:
            agent.learn(2)
        
        obs = obs_
        episode_reward += reward
    
    # 计算平均指标
    avg_rate = np.mean(episode_info['rates']) if episode_info['rates'] else 0
    avg_power = np.mean(episode_info['powers']) if episode_info['powers'] else 0
    avg_ul_rate = np.mean(episode_info['ul_rates']) if episode_info['ul_rates'] else 0
    avg_dl_rate = np.mean(episode_info['dl_rates']) if episode_info['dl_rates'] else 0
    
    return {
        'episode_reward': episode_reward,
        'avg_rate': avg_rate,
        'avg_power': avg_power,
        'avg_ul_rate': avg_ul_rate,
        'avg_dl_rate': avg_dl_rate
    }


def main():
    """主训练函数"""
    # 设置随机种子
    np.random.seed(42)
    
    # 创建日志记录器
    logger = TrainingLogger(LogConfig.LOG_FILE)
    
    # 记录开始时间
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logger.log(f"\n{'='*60}")
    logger.log(f"训练开始时间: {start_time_str}")
    logger.log(f"环境类型: {EnvironmentConfig.ENV_TYPE}")
    logger.log(f"UAV数量: {EnvironmentConfig.NUM_UAVS}")
    logger.log(f"用户数量: {len(EnvironmentConfig.UE_CLUSTER_1 + EnvironmentConfig.UE_CLUSTER_2)}")
    logger.log(f"{'='*60}\n")
    
    # 创建环境
    env = create_environment(EnvironmentConfig.ENV_TYPE)
    logger.log(f"✓ 环境创建成功: {env.__class__.__name__}")
    
    if hasattr(env, 'get_obs_dim'):
        logger.log(f"  观察维度: {env.get_obs_dim()}")
        logger.log(f"  动作维度: {env.get_action_dim()}")
    
    # 创建智能体
    agents = create_agents(env, TrainingConfig)
    logger.log(f"✓ 创建 {len(agents)} 个智能体")
    
    # 训练统计
    score_history = []
    
    # 训练循环
    logger.log(f"\n开始训练...")
    
    # 使用tqdm显示进度
    if LogConfig.SHOW_PROGRESS:
        episode_iter = tqdm(range(TrainingConfig.N_EPISODES), desc="Training")
    else:
        episode_iter = range(TrainingConfig.N_EPISODES)
    
    for episode in episode_iter:
        # 训练一个回合
        episode_data = train_episode(
            env, agents, 
            TrainingConfig.TIMESTAMP, 
            TrainingConfig.ACTION_BOUND
        )
        
        # 更新统计
        score_history.append(episode_data['episode_reward'])
        avg_score = np.mean(score_history[-100:]) / TrainingConfig.TIMESTAMP
        
        # 记录数据
        log_data = {
            'episode': episode,
            'total_reward': episode_data['episode_reward'],
            'avg_score': avg_score,
            'total_rate': episode_data['avg_rate'],
            'ul_rate': episode_data['avg_ul_rate'],
            'dl_rate': episode_data['avg_dl_rate'],
            'total_power': episode_data['avg_power']
        }
        
        # 打印进度
        if episode % TrainingConfig.PRINT_INTERVAL == 0 and episode > 0:
            if EnvironmentConfig.ENV_TYPE == 'dude':
                msg = (f"Episode {episode:4d} | "
                      f"Avg Score: {avg_score:7.2f} | "
                      f"Reward: {episode_data['episode_reward']:8.2f} | "
                      f"Rate: {episode_data['avg_rate']:6.2f} Mbps | "
                      f"Power: {episode_data['avg_power']:6.2f} W")
            else:
                msg = (f"Episode {episode:4d} | "
                      f"Avg Score: {avg_score:7.2f} | "
                      f"Reward: {episode_data['episode_reward']:8.2f}")
            
            if not LogConfig.SHOW_PROGRESS:
                logger.log(msg)
            else:
                tqdm.write(msg)
        
        logger.log_episode(episode, log_data)
        
        # 保存模型
        if episode % TrainingConfig.SAVE_INTERVAL == 0 and episode > 0:
            for i, agent in enumerate(agents):
                agent.save_models()
            if not LogConfig.SHOW_PROGRESS:
                logger.log(f"  ✓ 模型已保存 (Episode {episode})")
    
    # 记录结束时间
    end_time = time.time()
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    total_time = end_time - start_time
    
    logger.log(f"\n{'='*60}")
    logger.log(f"训练结束时间: {end_time_str}")
    logger.log(f"总训练时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
    logger.log(f"{'='*60}\n")
    
    # 绘制结果
    logger.log("生成训练结果图表...")
    logger.plot_results()
    logger.log(f"✓ 图表已保存:")
    logger.log(f"  - {LogConfig.AVG_REWARD_PLOT}")
    logger.log(f"  - {LogConfig.REWARD_PLOT}")
    if EnvironmentConfig.ENV_TYPE == 'dude':
        logger.log(f"  - {LogConfig.RATE_PLOT}")
    
    logger.log("\n训练完成！")


if __name__ == '__main__':
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 运行训练
    main()

