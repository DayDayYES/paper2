RL_Project/
│
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── setup.py                     # 安装配置（可选）
├── .gitignore                   # Git忽略文件
├── config/                      # 配置文件目录
│   ├── env_config.yaml         # 环境配置
│   ├── train_config.yaml       # 训练配置
│   └── model_config.yaml       # 模型配置
│
├── environments/                # 环境模块
│   ├── __init__.py
│   ├── base_env.py             # 基础环境类
│   ├── uav_env.py              # UAV环境实现
│   ├── uav_env_dude.py         # DUDe UAV环境
│   ├── multi_agent_env.py      # 多智能体环境包装
│   └── utils/                  # 环境工具函数
│       ├── observation_space.py
│       ├── action_space.py
│       └── reward_functions.py
│
├── algorithms/                  # 算法实现
│   ├── __init__.py
│   ├── matd3/                  # MATD3算法
│   │   ├── __init__.py
│   │   ├── actor.py            # Actor网络
│   │   ├── critic.py           # Critic网络
│   │   ├── replay_buffer.py    # 经验回放池
│   │   ├── noise.py            # 噪声策略（OU, Gaussian）
│   │   └── matd3_agent.py      # MATD3智能体类
│   │
│   ├── common/                 # 通用组件
│   │   ├── networks.py         # 神经网络基础结构
│   │   ├── utils.py            # 算法工具函数
│   │   └── logger.py           # 日志记录器
│   │
│   └── baselines/              # 基线算法（可选）
│       ├── dqn/
│       ├── ddpg/
│       └── td3/
│
├── scripts/                     # 运行脚本
│   ├── train.py                # 主训练脚本
│   ├── train_dude.py           # DUDe环境训练脚本
│   ├── evaluate.py             # 模型评估脚本
│   ├── visualize.py            # 结果可视化脚本
│   └── hyperparameter_tuning.py # 超参数调优
│
├── models/                      # 模型存储目录
│   ├── saved_models/           # 训练好的模型
│   │   ├── matd3_uav_1/       # 不同实验的模型
│   │   │   ├── actor_1.pth
│   │   │   ├── actor_2.pth
│   │   │   ├── critic_1.pth
│   │   │   └── critic_2.pth
│   │   └── matd3_uav_2/
│   │
│   └── checkpoints/            # 训练检查点
│       ├── episode_1000/
│       ├── episode_5000/
│       └── best_model/
│
├── results/                     # 实验结果
│   ├── logs/                   # 训练日志
│   │   ├── training_log.txt
│   │   ├── tensorboard/        # TensorBoard日志
│   │   └── metrics.json        # 评估指标
│   │
│   ├── figures/                # 生成图表
│   │   ├── training_curve.png
│   │   ├── reward_plot.png
│   │   └── policy_visualization.png
│   │
│   └── videos/                 # 训练过程录像
│       ├── episode_1.mp4
│       └── evaluation.mp4
│
├── tests/                       # 测试代码
│   ├── test_env.py             # 环境测试
│   ├── test_algorithm.py       # 算法测试
│   └── test_integration.py     # 集成测试
│
└── notebooks/                   # Jupyter notebooks（可选）
    ├── 01_environment_analysis.ipynb
    ├── 02_algorithm_testing.ipynb
    └── 03_results_visualization.ipynb