"""
MATD3训练脚本 - 混合动作空间的全双工UAV边缘计算环境
支持：离散用户调度 + 连续轨迹/功率/卸载控制
"""
import os
import time
import numpy as np
import torch
from tqdm import tqdm

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import EnvironmentConfig, TrainingConfig, LogConfig
from common.logger import TrainingLogger

# 导入环境
from environments.uav_env_dude import UAVEnvDUDe

# 导入算法
from algorithms.matd3.perddpg_torch import Agent


def create_environment():
    """
    创建全双工UAV环境
    
    Returns:
        env: 环境实例
    """
    env = UAVEnvDUDe(
        EnvironmentConfig.UE_CLUSTER_1,
        EnvironmentConfig.UE_CLUSTER_2,
        num_uavs=EnvironmentConfig.NUM_UAVS
    )
    return env


def create_agents(env, config):
    """
    创建智能体（处理混合动作空间）
    
    Args:
        env: 环境实例
        config: 训练配置
    
    Returns:
        agents: 智能体列表
    """
    agents = []
    n_agents = env.uav_num
    
    # 获取观察和动作维度
    obs_dim = env.get_obs_dim()
    continuous_action_dim = env.get_continuous_action_dim()  # 5维
    discrete_action_dim = env.get_discrete_action_dim()  # ue_num + 1
    
    # 总动作维度 = 连续动作 + 离散动作的logits
    total_action_dim = continuous_action_dim + discrete_action_dim
    
    for i in range(n_agents):
        agent = Agent(
            alpha=config.ALPHA,
            beta=config.BETA,
            input_dims=obs_dim,
            tau=config.TAU,
            n_actions=total_action_dim,  # 混合动作空间
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


def parse_action(action, env, already_selected):
    """
    解析混合动作
    
    Args:
        action: 网络输出的动作 [continuous_dim + discrete_dim]
        env: 环境实例
        already_selected: 已被其他UAV选择的用户列表
    
    Returns:
        continuous_action: 连续动作 [5]
        discrete_action: 离散动作（用户索引）
    """
    continuous_dim = env.get_continuous_action_dim()
    discrete_dim = env.get_discrete_action_dim()
    
    # 分离连续和离散部分
    continuous_action = action[:continuous_dim]
    discrete_logits = action[continuous_dim:]
    
    # 获取动作mask
    mask = env.get_action_mask(already_selected)
    
    # 应用mask并选择离散动作
    # 将不可选的动作设为很小的值
    masked_logits = discrete_logits.copy()
    masked_logits[mask == 0] = -1e10
    
    # 选择最大logit对应的动作
    discrete_action = np.argmax(masked_logits)
    
    return continuous_action, discrete_action


def train_episode(env, agents, config):
    """
    训练一个回合
    
    Args:
        env: 环境实例
        agents: 智能体列表
        config: 训练配置
    
    Returns:
        episode_data: 回合数据字典
    """
    state = env.reset()
    state_vec = env.get_state_vector()
    
    episode_reward = 0
    episode_info = {
        'delays': [],
        'energies': [],
        'scheduled_counts': []
    }
    
    done = False
    step = 0
    
    while not done and step < config.TIMESTAMP:
        # 选择动作
        all_actions = []
        discrete_actions = []
        continuous_actions = []
        already_selected = []
        
        for i, agent in enumerate(agents):
            # 获取网络输出
            action = agent.choose_action(state_vec)
            action = np.clip(action, -1, 1)
            all_actions.append(action)
            
            # 解析混合动作
            cont_act, disc_act = parse_action(action, env, already_selected)
            continuous_actions.append(cont_act)
            discrete_actions.append(disc_act)
            
            # 记录已选择的用户
            if disc_act < env.ue_num:
                already_selected.append(disc_act)
        
        # 执行动作
        next_state, reward, done, info = env.step(discrete_actions, continuous_actions)
        next_state_vec = env.get_state_vector()
        
        # 记录信息
        episode_info['delays'].append(info.get('total_delay', 0))
        episode_info['energies'].append(info.get('total_energy', 0))
        episode_info['scheduled_counts'].append(info.get('scheduled_count', 0))
        
        # 存储经验
        for i, agent in enumerate(agents):
            agent.remember((state_vec, all_actions[i], reward, next_state_vec, done))
        
        # 学习
        for agent in agents:
            agent.learn(2)
        
        state_vec = next_state_vec
        episode_reward += reward
        step += 1
    
    # 计算平均指标
    avg_delay = np.mean(episode_info['delays']) if episode_info['delays'] else 0
    avg_energy = np.mean(episode_info['energies']) if episode_info['energies'] else 0
    final_scheduled = episode_info['scheduled_counts'][-1] if episode_info['scheduled_counts'] else 0
    
    return {
        'episode_reward': episode_reward,
        'avg_delay': avg_delay,
        'avg_energy': avg_energy,
        'total_delay': sum(episode_info['delays']),
        'total_energy': sum(episode_info['energies']),
        'scheduled_count': final_scheduled,
        'steps': step
    }


def main():
    """主训练函数"""
    # 设置随机种子
    np.random.seed(TrainingConfig.RANDOM_SEED)
    torch.manual_seed(TrainingConfig.RANDOM_SEED)
    
    # 创建日志记录器
    logger = TrainingLogger(LogConfig.LOG_FILE)
    
    # 记录开始时间
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logger.log(f"\n{'='*60}")
    logger.log(f"训练开始时间: {start_time_str}")
    logger.log(f"{'='*60}\n")
    
    # 创建环境
    env = create_environment()
    logger.log(f"✓ 环境创建成功: {env.__class__.__name__}")
    logger.log(f"  UAV数量: {env.uav_num}")
    logger.log(f"  用户数量: {env.ue_num}")
    logger.log(f"  观察维度: {env.get_obs_dim()}")
    logger.log(f"  连续动作维度: {env.get_continuous_action_dim()}")
    logger.log(f"  离散动作维度: {env.get_discrete_action_dim()}")
    logger.log(f"  Episode最大步数: {env.max_steps}")
    
    # 启用CSV详细日志记录
    if LogConfig.ENABLE_CSV_LOGGING:
        env.enable_logging(log_dir='results/logs', filename='training_details.csv')
        logger.log(f"✓ CSV详细日志已启用")
    
    # 创建智能体
    agents = create_agents(env, TrainingConfig)
    logger.log(f"✓ 创建 {len(agents)} 个智能体")
    
    # 训练统计
    score_history = []
    delay_history = []
    energy_history = []
    
    # 训练循环
    logger.log(f"\n开始训练...")
    logger.log(f"总回合数: {TrainingConfig.N_EPISODES}")
    logger.log(f"每回合最大步数: {TrainingConfig.TIMESTAMP}")
    
    # 使用tqdm显示进度
    if LogConfig.SHOW_PROGRESS:
        episode_iter = tqdm(range(TrainingConfig.N_EPISODES), desc="Training")
    else:
        episode_iter = range(TrainingConfig.N_EPISODES)
    
    for episode in episode_iter:
        # 设置当前episode（用于CSV日志）
        env.set_episode(episode)
        
        # 训练一个回合
        episode_data = train_episode(env, agents, TrainingConfig)
        
        # 更新统计
        score_history.append(episode_data['episode_reward'])
        delay_history.append(episode_data['total_delay'])
        energy_history.append(episode_data['total_energy'])
        
        avg_score = np.mean(score_history[-100:])
        
        # 记录数据
        log_data = {
            'episode': episode,
            'total_reward': episode_data['episode_reward'],
            'avg_score': avg_score,
            'total_delay': episode_data['total_delay'],
            'avg_delay': episode_data['avg_delay'],
            'total_energy': episode_data['total_energy'],
            'avg_energy': episode_data['avg_energy'],
            'scheduled_count': episode_data['scheduled_count'],
            'steps': episode_data['steps']
        }
        
        # 打印进度
        if episode % TrainingConfig.PRINT_INTERVAL == 0 and episode > 0:
            msg = (f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:8.2f} | "
                  f"Reward: {episode_data['episode_reward']:8.2f} | "
                  f"Delay: {episode_data['total_delay']:6.3f}s | "
                  f"Energy: {episode_data['total_energy']:8.2f}J | "
                  f"Scheduled: {episode_data['scheduled_count']}/{env.ue_num}")
            
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
    
    # 关闭CSV日志
    if LogConfig.ENABLE_CSV_LOGGING:
        env.disable_logging()
        logger.log(f"✓ CSV详细日志已保存")
    
    # 绘制结果
    logger.log("生成训练结果图表...")
    logger.plot_results()
    logger.log(f"✓ 图表已保存到 results/ 目录")
    
    logger.log("\n训练完成！")


if __name__ == '__main__':
    # 设置环境变量
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 运行训练
    main()
