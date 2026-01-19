"""
全双工UAV边缘计算环境
实现：离散用户调度 + 连续轨迹/功率/卸载控制 + 全双工同时收发
新设计：
- 离散动作：每个UAV每时隙选择一个用户索引进行调度
- 连续动作：[速度, 水平角, 垂直角, 功率, 卸载比例]
- 全双工：UAV同时接收用户数据 + 向MBS发送部分卸载数据
- 优化目标：最小化时延 + 能耗（加权和）
"""
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from .uav import UAV, FUAV


class UAVEnvDUDe:
    """全双工UAV边缘计算环境类"""
    
    def __init__(self, c1, c2, num_uavs=2):
        """
        初始化环境
        
        Args:
            c1: 用户簇1的坐标列表
            c2: 用户簇2的坐标列表
            num_uavs: 旋翼无人机数量（默认2个）
        """
        # 用户设备相关
        self.cluster1 = c1
        self.cluster2 = c2
        self.uecord = self.cluster1 + self.cluster2
        self.ue_num = len(self.uecord)
        
        # 环境参数
        self.ground_width = 1200
        self.B = 10 * 10 ** 6  # 10MHz带宽（用户-UAV链路）
        self.B_back = 20 * 10 ** 6  # UAV-MBS链路带宽 20MHz（独立频段，无干扰）
        self.sigma = 10 ** (-10)  # 噪声功率 -70dBm
        
        # 通信环境参数 (dense urban)
        self.c1_param = 12.08  # eta_a
        self.c2_param = 0.11   # eta_b
        self.beta_los = 1  # eta_los
        self.beta_nlos = 20  # eta_nlos
        self.f_c = 2 * 10 ** 9  # 载波频率 2GHz
        self.c = 3 * 10 ** 8  # 光速
        
        # 用户设备参数
        self.P_ue_max = 1  # UE最大发射功率 (W) - 固定
        self.P_ue = self.P_ue_max
        
        # 全双工自干扰抑制系数
        self.xi = 100  # 自干扰抑制能力 (dB)，越大抑制越好
        self.xi_linear = 10 ** (self.xi / 10)  # 线性值
        
        # 无人机参数
        uav_params = {
            'f_uav': 1.2 * 10 ** 9,  # UAV计算频率 1.2GHz
            'P_uav_max': 2,  # 最大发射功率 2W
            'P_uav_min': 1,  # 最小发射功率 1W
            'B': self.B,
            'v_uav_min': 0,
            'v_uav_max': 60,
            'm_uav': 2,
            'C_uav': 200,
            'K_uav': 10 ** -28,  # 计算能耗系数 κ
            'n': 4,
            'rho': 1.225,
            'S_uav': 0.01,
            'g': 9.8,
            'c_r': 0.012,
            'c_t': 0.302,
            'A_r': 0.0314,
            's_r': 0.0955,
            'c_f': 0.131,
            'd_r': 0.834,
            'boundary': {
                'x_min': 0, 'x_max': 1000,
                'y_min': 0, 'y_max': 1000,
                'z_min': 200, 'z_max': 500
            }
        }
        
        # 保存UAV参数供后续使用
        self.f_uav = uav_params['f_uav']  # UAV计算频率
        self.K_uav = uav_params['K_uav']  # 计算能耗系数
        
        # 固定翼无人机参数（作为MBS/MEC服务器）
        fuav_params = uav_params.copy()
        fuav_params['f_fuav'] = 5 * 10 ** 9  # MBS计算频率 5GHz
        fuav_params['C_fuav'] = 300
        fuav_params['P_uav'] = 20  # MBS功率更大
        
        # 创建旋翼无人机
        self.uavs = []
        self.initial_positions = [
            [150, 150, 250],
            [950, 950, 250]
        ]
        for i in range(num_uavs):
            uav = UAV(uav_id=i, initial_position=self.initial_positions[i], uav_params=uav_params)
            self.uavs.append(uav)
        
        # 创建固定翼无人机（MBS/MEC服务器）
        self.mbs = FUAV(uav_id=0, fixed_position=[500, 500, 600], fuav_params=fuav_params)
        self.f_mbs = fuav_params['f_fuav']  # MBS计算频率 5GHz
        
        self.uav_num = len(self.uavs)
        
        # ==================== 用户调度相关 ====================
        # 已调度用户mask（0=未调度，1=已调度）
        self.scheduled_mask = np.zeros(self.ue_num, dtype=np.int32)
        # 当前时隙
        self.current_step = 0
        # Episode最大时隙数
        self.max_steps = int(np.ceil(self.ue_num / self.uav_num))
        
        # ==================== 权重系数 ====================
        self.w_delay = 0.5  # 时延权重
        self.w_energy = 0.5  # 能耗权重
        self.w_safety = 5.0  # 安全距离惩罚权重
        
        # 约束参数
        self.d_min = 100  # UAV间最小安全距离
        
        # 历史记录
        self.delay_history = []  # 时延历史
        self.energy_history = []  # 能耗历史
        
        # ==================== 任务模型参数 ====================
        self.task_D_in_min = 1 * 10 ** 6  # 输入数据大小最小值：1Mbits
        self.task_D_in_max = 2 * 10 ** 6  # 输入数据大小最大值：2Mbits
        self.task_D_out = 0.02 * 10 ** 6  # 输出数据大小：0.02Mbits（不考虑下行）
        
        # 任务类型和CPU周期数
        self.task_types = 5  # 任务类型数量
        self.task_omega_min = 10 ** 8  # CPU周期数最小值：10^8 cycles
        self.task_omega_max = 4 * 10 ** 8  # CPU周期数最大值：10^10 cycles
        
        # 任务时延约束
        self.task_tau_min = 0.1  # 最大时延最小值：0.1s
        self.task_tau_max = 5.0  # 最大时延最大值：5.0s
        
        # 任务存储：每个UE当前的任务
        self.tasks = [None] * self.ue_num
        
        # 生成初始任务
        self.generate_all_tasks()
        
        # ==================== CSV日志记录 ====================
        self.logging_enabled = False  # 是否启用日志记录
        self.log_file = None  # CSV文件路径
        self.csv_writer = None  # CSV写入器
        self.csv_file_handle = None  # 文件句柄
        self.current_episode = 0  # 当前episode编号
        
        # 初始化状态
        self.state = self._get_state()
    
    # ==================== CSV日志记录方法 ====================
    
    def enable_logging(self, log_dir='results/logs', filename='training_details.csv'):
        """
        启用CSV日志记录
        
        Args:
            log_dir: 日志目录
            filename: CSV文件名
        """
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # 创建日志目录
        full_log_dir = os.path.join(project_root, log_dir)
        os.makedirs(full_log_dir, exist_ok=True)
        
        # 完整文件路径
        self.log_file = os.path.join(full_log_dir, filename)
        
        # 创建CSV文件并写入表头
        self.csv_file_handle = open(self.log_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file_handle)
        
        # 写入表头
        headers = [
            # 基本信息
            'episode', 'step', 'uav_id', 'user_id',
            # UAV坐标
            'uav_x', 'uav_y', 'uav_z',
            # 用户坐标
            'ue_x', 'ue_y',
            # MBS坐标
            'mbs_x', 'mbs_y', 'mbs_z',
            # 距离
            'd_ue_uav', 'd_uav_mbs',
            # 任务参数
            'D_in_bits', 'omega_cycles',
            # 通信速率 (bps)
            'R_ue_uav_bps', 'R_uav_mbs_bps',
            # 功率和卸载比例
            'p_uav_W', 'lambda_offload',
            # 时延 (s)
            'T_trans_s', 'T_local_s', 'T_offload_link_s', 'T_mbs_compute_s', 'T_total_s',
            # 能耗 (J)
            'E_trans_J', 'E_comp_J', 'E_fly_J', 'E_total_J',
            # 奖励
            'step_reward', 'cumulative_reward'
        ]
        self.csv_writer.writerow(headers)
        self.csv_file_handle.flush()
        
        self.logging_enabled = True
        print(f"CSV logging enabled: {self.log_file}")
    
    def disable_logging(self):
        """关闭CSV日志记录"""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
        self.csv_writer = None
        self.logging_enabled = False
        print("CSV logging disabled")
    
    def set_episode(self, episode):
        """设置当前episode编号（用于日志记录）"""
        self.current_episode = episode
    
    def _log_step_detail(self, uav_idx, user_idx, uav_pos, ue_pos, mbs_pos,
                         d_ue_uav, d_uav_mbs, task, delay_info, energy_info,
                         p_uav, lambda_offload, T_total, E_total,
                         step_reward, cumulative_reward):
        """
        记录单步详细信息到CSV
        
        Args:
            uav_idx: UAV索引
            user_idx: 用户索引（-1表示空动作）
            uav_pos: UAV位置 [x, y, z]
            ue_pos: 用户位置 [x, y]（空动作时为None）
            mbs_pos: MBS位置 [x, y, z]
            d_ue_uav: UAV到用户距离
            d_uav_mbs: UAV到MBS距离
            task: 任务信息字典
            delay_info: 时延详细信息字典
            energy_info: 能耗详细信息字典
            p_uav: UAV发射功率
            lambda_offload: 卸载比例
            T_total: 总时延
            E_total: 总能耗
            step_reward: 当前步奖励
            cumulative_reward: 累积奖励
        """
        if not self.logging_enabled or self.csv_writer is None:
            return
        
        # 处理空动作情况
        if user_idx == -1 or ue_pos is None:
            ue_x, ue_y = -1, -1
            D_in, omega = 0, 0
            R_ue_uav, R_uav_mbs = 0, 0
            T_trans, T_local, T_offload_link, T_mbs_compute = 0, 0, 0, 0
            E_trans, E_comp = 0, 0
            d_ue_uav = -1
        else:
            ue_x, ue_y = ue_pos[0], ue_pos[1]
            D_in = task.get('D_in', 0) if task else 0
            omega = task.get('omega', 0) if task else 0
            R_ue_uav = delay_info.get('R_ue_uav', 0) if delay_info else 0
            R_uav_mbs = delay_info.get('R_uav_mbs', 0) if delay_info else 0
            T_trans = delay_info.get('T_trans', 0) if delay_info else 0
            T_local = delay_info.get('T_local', 0) if delay_info else 0
            T_offload_link = delay_info.get('T_offload_link', 0) if delay_info else 0
            T_mbs_compute = delay_info.get('T_mbs_compute', 0) if delay_info else 0
            E_trans = energy_info.get('E_trans_uav', 0) if energy_info else 0
            E_comp = energy_info.get('E_comp_uav', 0) if energy_info else 0
        
        E_fly = energy_info.get('E_fly', 0) if energy_info else 0
        
        # 写入一行数据
        row = [
            self.current_episode, self.current_step, uav_idx, user_idx,
            uav_pos[0], uav_pos[1], uav_pos[2],
            ue_x, ue_y,
            mbs_pos[0], mbs_pos[1], mbs_pos[2],
            d_ue_uav, d_uav_mbs,
            D_in, omega,
            R_ue_uav, R_uav_mbs,
            p_uav, lambda_offload,
            T_trans, T_local, T_offload_link, T_mbs_compute, T_total,
            E_trans, E_comp, E_fly, E_total,
            step_reward, cumulative_reward
        ]
        self.csv_writer.writerow(row)
        
        # 每10步刷新一次文件
        if self.current_step % 10 == 0:
            self.csv_file_handle.flush()
    
    # ==================== 状态空间 ====================
    
    def _get_state(self):
        """
        获取当前状态
        
        Returns:
            state: 状态字典，包含：
                - uav_positions: 所有UAV位置 [uav_num, 3]
                - mbs_position: MBS位置 [3]
                - ue_positions: 所有用户位置 [ue_num, 2] (地面用户，z=0)
                - scheduled_mask: 已调度用户mask [ue_num]
                - current_step: 当前时隙
        """
        state = {
            'uav_positions': np.array([uav.get_position() for uav in self.uavs]),
            'mbs_position': np.array(self.mbs.get_position()),
            'ue_positions': np.array(self.uecord),
            'scheduled_mask': self.scheduled_mask.copy(),
            'current_step': self.current_step
        }
        return state
    
    def get_state_vector(self):
        """
        获取扁平化的状态向量（用于神经网络输入）
        
        Returns:
            state_vec: 扁平化状态向量
        """
        state = self._get_state()
        
        # UAV位置 [uav_num * 3]
        uav_pos = state['uav_positions'].flatten()
        # MBS位置 [3]
        mbs_pos = state['mbs_position']
        # 用户位置 [ue_num * 2]
        ue_pos = state['ue_positions'].flatten()
        # 已调度mask [ue_num]
        mask = state['scheduled_mask'].astype(np.float32)
        
        # 拼接
        state_vec = np.concatenate([uav_pos, mbs_pos, ue_pos, mask])
        return state_vec
    
    def get_available_users(self):
        """
        获取当前可调度的用户列表
        
        Returns:
            available_users: 可调度用户索引列表
        """
        return np.where(self.scheduled_mask == 0)[0].tolist()
    
    def get_action_mask(self, already_selected=None):
        """
        获取动作mask（用于离散动作的合法性约束）
        
        Args:
            already_selected: 当前时隙已被其他UAV选择的用户索引列表
        
        Returns:
            mask: [ue_num + 1] 的mask数组
                  mask[i] = 1 表示用户i可选
                  mask[ue_num] = 1 表示"空动作"可选
        """
        if already_selected is None:
            already_selected = []
        
        mask = np.zeros(self.ue_num + 1, dtype=np.float32)
        
        # 未调度的用户可选（排除已被其他UAV选择的）
        available = self.get_available_users()
        for idx in available:
            if idx not in already_selected:
                mask[idx] = 1.0
        
        # 如果没有可选用户，则空动作可选
        if np.sum(mask[:self.ue_num]) == 0:
            mask[self.ue_num] = 1.0
        
        return mask
    
    # ==================== 距离和路径损耗计算 ====================
    
    def calculate_distance(self, pos1, pos2):
        """计算两点间的3D距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def calculate_uav_ue_distance(self, uav_idx, ue_idx):
        """计算UAV到UE的距离"""
        uav_pos = self.uavs[uav_idx].get_position()
        ue_pos = list(self.uecord[ue_idx]) + [0]  # UE在地面，z=0
        return self.calculate_distance(uav_pos, ue_pos)
    
    def calculate_uav_mbs_distance(self, uav_idx):
        """计算UAV到MBS的距离"""
        uav_pos = self.uavs[uav_idx].get_position()
        mbs_pos = self.mbs.get_position()
        return self.calculate_distance(uav_pos, mbs_pos)
    
    def compute_path_loss(self, distance, height):
        """
        计算路径损耗（包括LoS/NLoS概率）
        
        Args:
            distance: 3D距离
            height: UAV高度
        
        Returns:
            PL: 平均路径损耗（线性值，非dB）
        """
        # 仰角，确保 arcsin 的输入在 [-1, 1] 范围内
        sin_value = np.clip(height / (distance + 1e-10), -1.0, 1.0)
        theta = np.arcsin(sin_value) * 180 / np.pi
        
        # LoS概率
        P_los = 1 / (1 + self.c1_param * np.exp(-self.c2_param * (theta - self.c1_param)))
        P_nlos = 1 - P_los
        
        # 自由空间路径损耗（线性值）
        alpha = (4 * np.pi * self.f_c * distance / self.c) ** 2
        
        # LoS和NLoS路径损耗
        PL_los = alpha * (10 ** (self.beta_los / 10))
        PL_nlos = alpha * (10 ** (self.beta_nlos / 10))
        
        # 平均路径损耗
        PL = P_los * PL_los + P_nlos * PL_nlos
        
        return max(PL, 1e-10)  # 避免除零
    
    # ==================== 全双工速率计算 ====================
    
    def compute_uplink_rate_with_self_interference(self, uav_idx, ue_idx, p_uav):
        """
        计算全双工模式下的上行速率（考虑自干扰）
        
        全双工：UAV同时接收用户数据 + 向MBS发送卸载数据
        自干扰：UAV发射给MBS的信号干扰UAV接收用户的信号
        
        Args:
            uav_idx: UAV索引
            ue_idx: 用户索引
            p_uav: UAV发射功率（向MBS发送时的功率）
        
        Returns:
            R_ue_uav: 用户到UAV的上行速率 (bps)
        """
        # 计算用户到UAV的距离和路径损耗
        d_ue_uav = self.calculate_uav_ue_distance(uav_idx, ue_idx)
        uav_height = self.uavs[uav_idx].get_position()[2]
        PL_ue_uav = self.compute_path_loss(d_ue_uav, uav_height)
        
        # 接收信号功率
        P_rx = self.P_ue / PL_ue_uav
        
        # 自干扰功率（UAV发射功率 / 自干扰抑制系数）
        I_self = p_uav / self.xi_linear
        
        # SINR
        SINR = P_rx / (I_self + self.sigma)
        
        # 上行速率
        R_ue_uav = self.B * np.log2(1 + SINR)
        
        return R_ue_uav
    
    def compute_uav_mbs_rate(self, uav_idx, p_uav):
        """
        计算UAV到MBS的传输速率（独立频段，无干扰）
        
        Args:
            uav_idx: UAV索引
            p_uav: UAV发射功率
        
        Returns:
            R_uav_mbs: UAV到MBS的速率 (bps)
        """
        # 计算UAV到MBS的距离和路径损耗
        d_uav_mbs = self.calculate_uav_mbs_distance(uav_idx)
        uav_height = self.uavs[uav_idx].get_position()[2]
        mbs_height = self.mbs.get_position()[2]
        avg_height = (uav_height + mbs_height) / 2
        PL_uav_mbs = self.compute_path_loss(d_uav_mbs, avg_height)
        
        # SINR（独立频段，无干扰）
        SINR = (p_uav / PL_uav_mbs) / self.sigma
        
        # 速率
        R_uav_mbs = self.B_back * np.log2(1 + SINR)
        
        return R_uav_mbs
    
    # ==================== 时延计算 ====================
    
    def compute_task_delay(self, uav_idx, ue_idx, p_uav, lambda_offload):
        """
        计算任务完成时延（全双工并行模型）
        
        时延模型：
        T_trans = D_in / R_ue_uav                    # 用户→UAV传输时延
        T_local = (1-λ) * omega / f_uav              # UAV本地计算时延
        T_offload_link = λ * D_in / R_uav_mbs        # UAV→MBS传输时延（与T_trans并行）
        T_mbs_compute = λ * omega / f_mbs            # MBS计算时延
        
        T_total = max(T_trans + T_local, T_offload_link + T_mbs_compute)
        
        Args:
            uav_idx: UAV索引
            ue_idx: 用户索引
            p_uav: UAV发射功率
            lambda_offload: 卸载比例 [0, 1]
        
        Returns:
            T_total: 总时延 (s)
            delay_info: 时延详细信息字典
        """
        # 获取任务信息
        task = self.tasks[ue_idx]
        if task is None:
            return 0.0, {}
        
        D_in = task['D_in']  # 输入数据大小 (bits)
        omega = task['omega']  # CPU周期数 (cycles)
        
        # 计算速率
        R_ue_uav = self.compute_uplink_rate_with_self_interference(uav_idx, ue_idx, p_uav)
        R_uav_mbs = self.compute_uav_mbs_rate(uav_idx, p_uav)
        
        # 确保速率有效
        R_ue_uav = max(R_ue_uav, 1e-10)
        R_uav_mbs = max(R_uav_mbs, 1e-10)
        
        # 用户→UAV传输时延
        T_trans = D_in / R_ue_uav
        
        # UAV本地计算时延  0.5 * 100~400M / 1.2G
        T_local = (1 - lambda_offload) * omega / self.f_uav
        
        # UAV→MBS传输时延（全双工，与T_trans并行）
        T_offload_link = lambda_offload * D_in / R_uav_mbs
        
        # MBS计算时延  0.5 * 100~400M / 5G
        T_mbs_compute = lambda_offload * omega / self.f_mbs
        
        # 总时延（并行模型）
        T_local_path = T_trans + T_local  # 本地计算路径
        T_offload_path = T_offload_link + T_mbs_compute  # 卸载路径
        T_total = max(T_local_path, T_offload_path)
        
        delay_info = {
            'T_trans': T_trans,
            'T_local': T_local,
            'T_offload_link': T_offload_link,
            'T_mbs_compute': T_mbs_compute,
            'T_local_path': T_local_path,
            'T_offload_path': T_offload_path,
            'R_ue_uav': R_ue_uav,
            'R_uav_mbs': R_uav_mbs
        }
        
        return T_total, delay_info
    
    # ==================== 能耗计算 ====================
    
    def compute_task_energy(self, uav_idx, ue_idx, p_uav, lambda_offload, fly_energy=0):
        """
        计算任务处理能耗（UAV侧）
        
        能耗模型：
        E_trans_uav = P_uav * T_offload_link         # UAV传输能耗
        E_comp_uav = κ * f_uav² * (1-λ) * omega      # UAV计算能耗
        E_fly = 飞行能耗
        
        Args:
            uav_idx: UAV索引
            ue_idx: 用户索引
            p_uav: UAV发射功率
            lambda_offload: 卸载比例 [0, 1]
            fly_energy: 飞行能耗 (J)
        
        Returns:
            E_total: 总能耗 (J)
            energy_info: 能耗详细信息字典
        """
        # 获取任务信息
        task = self.tasks[ue_idx]
        if task is None:
            return fly_energy, {'E_fly': fly_energy}
        
        D_in = task['D_in']
        omega = task['omega']
        
        # 计算速率（用于计算传输时间）
        R_uav_mbs = self.compute_uav_mbs_rate(uav_idx, p_uav)
        R_uav_mbs = max(R_uav_mbs, 1e-10)
        
        # UAV→MBS传输时延
        T_offload_link = lambda_offload * D_in / R_uav_mbs
        
        # UAV传输能耗
        E_trans_uav = p_uav * T_offload_link
        
        # UAV计算能耗: E = κ * f² * cycles
        # 注意：这里用 f² 而不是 f³，因为 P = κ * f³，E = P * t = κ * f³ * (cycles/f) = κ * f² * cycles
        E_comp_uav = self.K_uav * (self.f_uav ** 2) * (1 - lambda_offload) * omega
        
        # 总能耗
        E_total = E_trans_uav + E_comp_uav
        
        energy_info = {
            'E_trans_uav': E_trans_uav,
            'E_comp_uav': E_comp_uav,
            'E_fly': fly_energy,
            'T_offload_link': T_offload_link
        }
        
        return E_total, energy_info
    
    # ==================== 飞行能耗 ====================
    
    def compute_fly_energy(self, d, theta_vertical, uav):
        """计算UAV飞行能耗"""
        if d < 1e-6:  # 几乎没有移动
            return 0.0
        
        a = 5
        theta_c = 0.5 * np.pi - theta_vertical
        F = np.sqrt(((uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.cos(theta_c)) ** 2 +
                    (-1 * (uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.sin(theta_c) - 
                     uav.m_uav * uav.g) ** 2) / uav.n
        
        term1 = (uav.c_r / 8) * (F / (uav.c_t * uav.A_r * uav.rho) + 3 * d**2) * \
                np.sqrt(F * uav.rho * uav.s_r ** 2 * uav.A_r / uav.c_t)
        term2 = (1 + uav.c_f) * F * np.sqrt(np.sqrt((F ** 2 / (4 * uav.rho ** 2 * uav.A_r ** 2)) + 
                                                      (d ** 4 / 4)) - d ** 2 / 2)
        term3 = 0.5 * uav.d_r * d ** 3 * uav.rho * uav.s_r * uav.A_r + \
                uav.m_uav * uav.g * d / uav.n * (np.sin(theta_c))
        P_fly = uav.n * (term1 + term2 + term3)
        
        return max(P_fly, 0.0)
    
    # ==================== 任务生成 ====================
    
    def generate_task(self, ue_index):
        """
        为指定UE生成一个新任务
        
        Args:
            ue_index: UE索引
        
        Returns:
            task: 任务字典
        """
        task = {
            'D_in': np.random.uniform(self.task_D_in_min, self.task_D_in_max),
            'D_out': self.task_D_out,
            'omega': np.random.uniform(self.task_omega_min, self.task_omega_max),
            's': np.random.randint(0, self.task_types),
            'tau': np.random.uniform(self.task_tau_min, self.task_tau_max)
        }
        return task
    
    def generate_all_tasks(self):
        """为所有UE生成新任务"""
        for i in range(self.ue_num):
            self.tasks[i] = self.generate_task(i)
    
    def get_task_info(self, ue_index):
        """获取指定UE的任务信息"""
        if ue_index < len(self.tasks):
            return self.tasks[ue_index]
        return None
    
    # ==================== 环境step ====================
    
    def step(self, discrete_actions, continuous_actions):
        """
        环境执行一步
        
        Args:
            discrete_actions: 每个UAV的离散动作（用户索引）
                             [uav_num] 维度，值域 [0, ue_num]
                             ue_num 表示"空动作"（不服务任何用户）
            continuous_actions: 每个UAV的连续动作
                               [uav_num, 5] 维度
                               每个UAV: [v_norm, theta_h_norm, theta_v_norm, p_norm, lambda_norm]
                               所有值在 [-1, 1] 范围内
        
        Returns:
            state: 新状态
            reward: 奖励（标量，所有UAV共享）
            done: 是否结束
            info: 额外信息字典
        """
        info = {}
        total_delay = 0.0
        total_energy = 0.0
        reward = 0.0
        
        # 检查离散动作的合法性（不允许两个UAV选同一用户）
        selected_users = []
        for uav_idx, user_idx in enumerate(discrete_actions):
            if user_idx < self.ue_num:  # 不是空动作
                if user_idx in selected_users:
                    # 非法动作：重复选择，给惩罚
                    reward -= 50.0
                    print(f"Warning: UAV {uav_idx} selected already chosen user {user_idx}")
                elif self.scheduled_mask[user_idx] == 1:
                    # 非法动作：选择已调度用户，给惩罚
                    reward -= 50.0
                    print(f"Warning: UAV {uav_idx} selected already scheduled user {user_idx}")
                else:
                    selected_users.append(user_idx)
        
        # 处理每个UAV的动作
        uav_delays = []
        uav_energies = []
        fly_energy_list = []
        
        for uav_idx in range(self.uav_num):
            uav = self.uavs[uav_idx]
            
            # 解析连续动作（从[-1,1]归一化到[0,1]）
            cont_action = continuous_actions[uav_idx]
            v_norm = (cont_action[0] + 1) / 2  # 速度
            theta_h_norm = (cont_action[1] + 1) / 2  # 水平角
            theta_v_norm = (cont_action[2] + 1) / 2  # 垂直角
            p_norm = (cont_action[3] + 1) / 2  # 功率
            lambda_norm = (cont_action[4] + 1) / 2  # 卸载比例
            
            # 1. UAV移动
            move_action = np.array([v_norm, theta_h_norm, theta_v_norm])
            success, old_pos, d, theta_v = uav.move(move_action)
            
            # 计算飞行能耗
            E_fly = self.compute_fly_energy(d, theta_v, uav)
            fly_energy_list.append(E_fly)
            
            # 边界惩罚
            if not success:
                reward -= 10.0
            
            # 2. 解析功率和卸载比例
            p_uav = uav.P_uav_min + p_norm * (uav.P_uav_max - uav.P_uav_min)  # 实际发射功率
            lambda_offload = lambda_norm  # 卸载比例 [0, 1]
            
            # 3. 处理离散动作（用户调度）
            user_idx = discrete_actions[uav_idx]
            
            # 获取位置信息（用于日志）
            uav_pos = uav.get_position()
            mbs_pos = self.mbs.get_position()
            d_uav_mbs = self.calculate_uav_mbs_distance(uav_idx)
            
            if user_idx < self.ue_num and user_idx in selected_users:
                # 有效调度
                # 标记用户已调度
                self.scheduled_mask[user_idx] = 1
                
                # 获取用户位置和距离
                ue_pos = self.uecord[user_idx]
                d_ue_uav = self.calculate_uav_ue_distance(uav_idx, user_idx)
                task = self.tasks[user_idx]
                
                # 计算时延
                T_total, delay_info = self.compute_task_delay(uav_idx, user_idx, p_uav, lambda_offload)
                
                # 计算能耗（包含飞行能耗）
                E_total, energy_info = self.compute_task_energy(uav_idx, user_idx, p_uav, lambda_offload, E_fly)
                
                uav_delays.append(T_total)
                uav_energies.append(E_total)
                
                # 记录到info
                info[f'uav_{uav_idx}'] = {
                    'user_idx': user_idx,
                    'delay': T_total,
                    'energy': E_total,
                    'lambda': lambda_offload,
                    'power': p_uav,
                    'delay_info': delay_info,
                    'energy_info': energy_info
                }
            else:
                # 空动作或无效动作
                ue_pos = None
                d_ue_uav = -1
                task = None
                T_total = 0.0
                delay_info = {}
                energy_info = {'E_fly': E_fly}
                E_total = E_fly
                
                uav_delays.append(0.0)
                uav_energies.append(E_fly)  # 只有飞行能耗
                
                info[f'uav_{uav_idx}'] = {
                    'user_idx': -1,  # 表示没有服务用户
                    'delay': 0.0,
                    'energy': E_fly,
                    'lambda': 0.0,
                    'power': p_uav
                }
            
            # CSV日志记录（临时存储，等计算完reward后再写入）
            info[f'uav_{uav_idx}']['_log_data'] = {
                'uav_pos': uav_pos,
                'ue_pos': ue_pos,
                'mbs_pos': mbs_pos,
                'd_ue_uav': d_ue_uav,
                'd_uav_mbs': d_uav_mbs,
                'task': task,
                'delay_info': delay_info,
                'energy_info': energy_info,
                'p_uav': p_uav,
                'lambda_offload': lambda_offload,
                'T_total': T_total,
                'E_total': E_total
            }
        
        # 计算总时延和能耗
        total_delay = sum(uav_delays)
        total_energy = sum(uav_energies)
        
        # UAV间安全距离惩罚
        safety_penalty = 0.0
        for i in range(self.uav_num):
            for j in range(i + 1, self.uav_num):
                dist = np.linalg.norm(self.uavs[i].position - self.uavs[j].position)
                if dist < self.d_min:
                    safety_penalty += (self.d_min - dist) / self.d_min
        
        # 计算奖励（最小化时延和能耗）
        # 归一化处理
        delay_normalized = total_delay  # 时延约0.1~5s
        energy_normalized = total_energy / 1000  # 能耗归一化（除以1000）
        
        reward += -(self.w_delay * delay_normalized + 
                    self.w_energy * energy_normalized +
                    self.w_safety * safety_penalty)
        
        # 检查并处理 NaN 或无穷大
        if not np.isfinite(reward):
            reward = -100.0
            print(f"Warning: Invalid reward. delay={total_delay:.4f}, energy={total_energy:.4f}")
        
        # CSV日志记录（在计算完reward后写入）
        if self.logging_enabled:
            cumulative_reward = reward  # 当前步的累积奖励
            for uav_idx in range(self.uav_num):
                log_data = info[f'uav_{uav_idx}'].get('_log_data', {})
                user_idx = info[f'uav_{uav_idx}']['user_idx']
                
                self._log_step_detail(
                    uav_idx=uav_idx,
                    user_idx=user_idx,
                    uav_pos=log_data.get('uav_pos', [0, 0, 0]),
                    ue_pos=log_data.get('ue_pos'),
                    mbs_pos=log_data.get('mbs_pos', [0, 0, 0]),
                    d_ue_uav=log_data.get('d_ue_uav', -1),
                    d_uav_mbs=log_data.get('d_uav_mbs', 0),
                    task=log_data.get('task'),
                    delay_info=log_data.get('delay_info', {}),
                    energy_info=log_data.get('energy_info', {}),
                    p_uav=log_data.get('p_uav', 0),
                    lambda_offload=log_data.get('lambda_offload', 0),
                    T_total=log_data.get('T_total', 0),
                    E_total=log_data.get('E_total', 0),
                    step_reward=reward,
                    cumulative_reward=cumulative_reward
                )
                # 清理临时数据
                if '_log_data' in info[f'uav_{uav_idx}']:
                    del info[f'uav_{uav_idx}']['_log_data']
        
        # 更新时隙
        self.current_step += 1
        
        # 记录历史
        self.delay_history.append(total_delay)
        self.energy_history.append(total_energy)
        
        # 检查是否结束（所有用户都已调度）
        done = (np.sum(self.scheduled_mask) == self.ue_num) or (self.current_step >= self.max_steps)
        
        # 更新状态
        self.state = self._get_state()
        
        # 汇总信息
        info['total_delay'] = total_delay
        info['total_energy'] = total_energy
        info['safety_penalty'] = safety_penalty
        info['scheduled_count'] = int(np.sum(self.scheduled_mask))
        info['remaining_users'] = self.ue_num - int(np.sum(self.scheduled_mask))
        info['current_step'] = self.current_step
        info['done'] = done
        
        return self.state, reward, done, info
    
    def reset(self):
        """重置环境"""
        # 重置UAV位置
        for i, uav in enumerate(self.uavs):
            uav.reset(self.initial_positions[i])
        
        # 重置调度状态
        self.scheduled_mask = np.zeros(self.ue_num, dtype=np.int32)
        self.current_step = 0
        
        # 重置历史
        self.delay_history = []
        self.energy_history = []
        
        # 生成新任务
        self.generate_all_tasks()
        
        # 更新状态
        self.state = self._get_state()
        
        return self.state
    
    # ==================== 动作空间维度 ====================
    
    def get_continuous_action_dim(self):
        """
        获取连续动作空间维度
        
        Returns:
            dim: 每个UAV的连续动作维度 = 5
                 [速度, 水平角, 垂直角, 功率, 卸载比例]
        """
        return 5
    
    def get_discrete_action_dim(self):
        """
        获取离散动作空间维度
        
        Returns:
            dim: 离散动作数量 = ue_num + 1
                 [0, ue_num-1] 表示选择用户索引
                 ue_num 表示"空动作"
        """
        return self.ue_num + 1
    
    def get_obs_dim(self):
        """
        获取观察空间维度
        
        Returns:
            dim: 状态向量维度
                 = uav_num * 3 (UAV位置)
                 + 3 (MBS位置)
                 + ue_num * 2 (用户位置)
                 + ue_num (已调度mask)
        """
        return self.uav_num * 3 + 3 + self.ue_num * 2 + self.ue_num
    
    # ==================== 辅助方法 ====================
    
    def render(self, mode='human'):
        """可视化当前环境状态"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制用户
        for i, ue_pos in enumerate(self.uecord):
            if self.scheduled_mask[i] == 1:
                color = 'green'  # 已调度
                marker = 's'
            else:
                color = 'blue'  # 未调度
                marker = 'o'
            ax.scatter(ue_pos[0], ue_pos[1], c=color, marker=marker, s=100, label=f'UE{i}' if i == 0 else '')
            ax.annotate(f'{i}', (ue_pos[0], ue_pos[1]), textcoords="offset points", xytext=(5, 5))
        
        # 绘制UAV
        for i, uav in enumerate(self.uavs):
            pos = uav.get_position()
            ax.scatter(pos[0], pos[1], c='red', marker='^', s=200, label=f'UAV{i}')
            ax.annotate(f'UAV{i}\nz={pos[2]:.0f}', (pos[0], pos[1]), textcoords="offset points", xytext=(10, 10))
        
        # 绘制MBS
        mbs_pos = self.mbs.get_position()
        ax.scatter(mbs_pos[0], mbs_pos[1], c='purple', marker='*', s=300, label='MBS')
        ax.annotate(f'MBS\nz={mbs_pos[2]:.0f}', (mbs_pos[0], mbs_pos[1]), textcoords="offset points", xytext=(10, 10))
        
        ax.set_xlim(0, self.ground_width)
        ax.set_ylim(0, self.ground_width)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'UAV Environment - Step {self.current_step}/{self.max_steps}\n'
                     f'Scheduled: {np.sum(self.scheduled_mask)}/{self.ue_num}')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig
