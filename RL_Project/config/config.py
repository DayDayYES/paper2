"""
配置文件 - 训练超参数和环境设置
"""

# ==================== 环境配置 ====================
class EnvironmentConfig:
    """环境配置"""
    # 用户簇配置
    UE_CLUSTER_1 = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], 
                    [352.51, 636.99, 0], [365.74, 971.82, 0], [320.80, 406.66, 0], 
                    [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], 
                    [267.70, 926.15, 0]]
    
    UE_CLUSTER_2 = [[966.09, 757.82, 0], [304.61, 786.76, 0], [427.41, 1158.40, 0], 
                    [861.97, 925.30, 0], [541.30, 815.78, 0], [413.33, 1023.47, 0], 
                    [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], 
                    [878.19, 930.27, 0]]
    
    # 无人机数量
    NUM_UAVS = 2
    
    # 环境类型选择
    # 'legacy': 原始环境（计算卸载）
    # 'dude': 解耦关联的全双工环境
    ENV_TYPE = 'dude'


# ==================== 训练配置 ====================
class TrainingConfig:
    """训练超参数配置"""
    # 训练参数
    N_EPISODES = 800          # 训练回合数
    TIMESTAMP = 100           # 每回合的时间步数
    BATCH_SIZE = 64          # 批大小
    MEMORY_SIZE = 1000000    # 经验回放缓冲区大小
    
    # 学习率
    ALPHA = 0.0001           # Actor学习率
    BETA = 0.001             # Critic学习率
    GAMMA = 0.99             # 折扣因子
    TAU = 0.005              # 软更新系数
    
    # 探索噪声
    NOISE = 0.2              # 动作噪声标准差
    
    # 网络结构
    CRITIC_FC1_DIMS = 512
    CRITIC_FC2_DIMS = 256
    CRITIC_FC3_DIMS = 256
    ACTOR_FC1_DIMS = 1024
    ACTOR_FC2_DIMS = 512
    
    # 策略延迟更新
    POLICY_DELAY = 2
    
    # 打印和保存
    PRINT_INTERVAL = 1       # 打印间隔
    SAVE_INTERVAL = 100      # 保存模型间隔
    
    # 动作边界
    ACTION_BOUND = [-1, 1]


# ==================== 日志配置 ====================
class LogConfig:
    """日志和输出配置"""
    # 输出文件名
    LOG_FILE = 'training_log_dude.txt'
    REWARD_PLOT = 'Reward_dude.png'
    AVG_REWARD_PLOT = 'Avg_reward_dude.png'
    RATE_PLOT = 'Rate_dude.png'
    
    # 是否保存详细日志
    SAVE_DETAILED_LOG = True
    
    # 是否显示进度条
    SHOW_PROGRESS = True

