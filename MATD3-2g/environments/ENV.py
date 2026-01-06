import math
import random

import matplotlib.pyplot as plt
import numpy as np
from uav import UAV, FUAV


class Environment():
	"""无人机任务环境类 - 管理UAV、UE和环境交互"""

	def __init__(self, c1, c2):
		"""
		初始化环境
		
		Args:
			c1: 用户簇1的坐标列表
			c2: 用户簇2的坐标列表
		"""
		# 初始化matplotlib图形
		self.k = plt.subplots()
		self.fig = self.k[0]
		self.ax = self.k[1]
		
		# 用户设备相关
		self.cluster1 = c1
		self.cluster2 = c2
		self.uecord = self.cluster1 + self.cluster2
		self.ue_num = len(self.uecord)
		self.connected_array = np.full(self.ue_num, -1)
		
		# 用户设备参数
		self.P_ue = 0.1  # 用户设备的传输功率 (W)
		self.f_ue = 0.2 * 10 ** 9  # 用户设备的计算频率
		self.C_ue = 300  # 1bit数据所需的CPU周期数
		self.v_ue = 1  # 用户移动速度
		
		# 环境参数
		self.ground_width = 1200
		self.B = 10 * 10 ** 6  # 10MHz的带宽
		self.sigma = 10 ** (-10)  # 噪声功率-70dBm
		self.w = 0.0001  # 权重系数
		
		# 通信环境参数 (dense urban)
		self.eta_a = 12.08
		self.eta_los = 1
		self.eta_b = 0.11
		self.eta_nlos = 20
		self.beta_0 = 10 ** -5  # -50db
		self.p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm
		self.p_noisy_nlos = 10 ** (-11)  # 噪声功率-80dBm
		self.f_c = 2 * 10 ** 9  # 系统频率
		self.v_c = 3 * 10 ** 8  # 光速
		self.capacity_min = 4 * 10 ** 6  # 最小容量要求 4Mbps
		
		# 任务相关
		self.block_flag_list = np.random.randint(0, 2, self.ue_num)
		self.task_list = np.random.randint(2021440, 3045729, self.ue_num)
		
		# 无人机参数字典
		uav_params = {
			'f_uav': 1.2 * 10 ** 9,
			'P_uav': 5,
			'B': self.B,
			'v_uav_min': 0,
			'v_uav_max': 60,
			'm_uav': 2,
			'C_uav': 200,
			'K_uav': 10 ** -28,
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
				'x_min': 200, 'x_max': 1150,
				'y_min': 200, 'y_max': 1150,
				'z_min': 200, 'z_max': 500
			}
		}
		
		# 固定翼无人机参数
		fuav_params = uav_params.copy()
		fuav_params['f_fuav'] = 3 * 10 ** 9
		fuav_params['C_fuav'] = 300
		
		# 创建无人机实例
		self.uavs = []
		initial_positions = [
			[350, 350, 250],
			[850, 850, 250],
			[350, 850, 250]
		]
		for i, pos in enumerate(initial_positions):
			uav = UAV(uav_id=i, initial_position=pos, uav_params=uav_params)
			self.uavs.append(uav)
		
		# 创建固定翼无人机
		self.fuav = FUAV(uav_id=99, fixed_position=[400, 400, 300], fuav_params=fuav_params)
		
		self.uav_num = len(self.uavs)
		self.state = [uav.get_position() for uav in self.uavs]
		
		# 公平性历史记录
		self.cover_history_ue = []
		self.cover_history_uav = []

	# ==================== 通信模型相关方法 ====================
	
	def capacity(self, d, uav_location):
		"""
		计算信道容量
		
		Args:
			d: 距离矩阵 [uav_num, ue_num]
			uav_location: 无人机位置列表 [uav_num, 3]
		
		Returns:
			L_matrix: 路径损耗矩阵 [uav_num, ue_num]
			capacity_matrix: 信道容量矩阵 [uav_num, ue_num]
		"""
		def capacity_single_distance(distance, uav_height):
			theta = 180 * np.arcsin(uav_height / distance) / np.pi
			p_los = 1 / (1 + self.eta_a * np.exp(-self.eta_b * (theta - self.eta_a)))
			p_nlos = 1 - p_los
			L_fs = 20 * np.log10(distance) + 20 * np.log10(self.f_c) + 20 * np.log10(4 * np.pi / self.v_c)
			L_los = L_fs + self.eta_los
			L_nlos = L_fs + self.eta_nlos
			L = p_los * L_los + p_nlos * L_nlos
			gain = 1 / (10 ** (L / 10))
			SNR = self.P_ue * gain / self.sigma
			capacity = self.B * np.log2(1 + SNR)
			return L, capacity

		# 使用numpy的vectorize函数将capacity_single_distance函数应用到d矩阵的每个元素上
		capacity_vectorized = np.vectorize(capacity_single_distance)
		uav_locations = np.array(uav_location)
		uav_heights = uav_locations[:, 2]
		uav_heights = uav_heights.reshape(-1, 1)
		L_matrix, capacity_matrix = capacity_vectorized(d, uav_heights)
		return L_matrix, capacity_matrix


	def coverge(self, spatial_distances, capacity):
		"""
		计算覆盖情况
		
		Args:
			spatial_distances: 空间距离矩阵 [uav_num, ue_num]
			capacity: 信道容量矩阵 [uav_num, ue_num]
		
		Returns:
			coverage: 覆盖矩阵 [uav_num, ue_num] (0或1)
			cover1: 用户连接的无人机ID [ue_num]
		"""
		# 找到每个用户与其最近的无人机的距离
		min_distances = np.min(spatial_distances, axis=0)
		# 创建一个新的矩阵，该矩阵的每个元素表示对应的无人机是否是最近的无人机
		is_nearest_uav = np.where(spatial_distances == min_distances, 1, 0)
		# 创建覆盖矩阵
		cover = np.where((capacity >= self.capacity_min) & (is_nearest_uav == 1), 1, 0)
		coverage = np.array(cover).reshape(self.uav_num, self.ue_num)

		# 初始化覆盖矩阵
		cover1 = np.zeros((self.uav_num, self.ue_num))
		# 遍历每个无人机
		for i in range(self.uav_num):
			# 生成当前无人机的覆盖矩阵
			cover_i = np.where((capacity[i, :] >= self.capacity_min) & (is_nearest_uav[i, :] == 1), i + 1, 0)
			# 将当前无人机的覆盖矩阵加到总覆盖矩阵上
			cover1 += cover_i
		cover1 = np.sum(cover1, axis=0) / self.uav_num
		cover1 = cover1.astype(int)
		
		return coverage, cover1

	def calculate_distances(self, uav_locations, user_locations):
		"""
		计算距离矩阵
		
		Args:
			uav_locations: 无人机位置列表
			user_locations: 用户位置列表
		
		Returns:
			spatial_distances: 空间距离矩阵 [uav_num, ue_num]
			uav_fix_distances: 无人机到固定翼无人机的距离 [uav_num, 1]
		"""
		uav_fix_location = self.fuav.get_position()
		distances = []
		for uav_location in uav_locations:
			uav_location = np.array(uav_location)
			for user_location in user_locations:
				user_location = np.array(user_location)
				# 计算空间距离
				spatial_distance = np.sqrt(np.sum((uav_location - user_location) ** 2))
				# 计算无人机和固定翼无人机的空间距离
				uav_fix_distance = np.sqrt(np.sum((uav_location - uav_fix_location) ** 2))
				distances.append((spatial_distance, uav_fix_distance))
		spatial_distances, uav_fix_distances = zip(*distances)
		spatial_distances = np.array(spatial_distances).reshape(self.uav_num, self.ue_num)
		uav_fix_distances = np.array(uav_fix_distances).reshape(self.uav_num, self.ue_num)
		uav_fix_distances = np.mean(uav_fix_distances, axis=1).reshape(self.uav_num, 1)
		return spatial_distances, uav_fix_distances

	# ==================== 能耗模型相关方法 ====================
	
	def fly_energy(self, d, theta_vertical, uav):
		"""
		计算无人机的飞行能耗
		
		Args:
			d: 飞行距离 (m)
			theta_vertical: 垂直偏转角 (rad)
			uav: UAV对象
		
		Returns:
			P_fly: 飞行功率 (W)
		"""
		a = 5  # 加速度
		theta_c = 0.5 * np.pi - theta_vertical
		F = np.sqrt(((uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.cos(theta_c)) ** 2 +
		            (-1 * (uav.m_uav * a + 0.5 * uav.rho * d ** 2 * uav.S_uav) * d * np.sin(theta_c) - uav.m_uav * uav.g) ** 2) / uav.n

		# 计算无人机的升力
		term1 = (uav.c_r / 8) * (F / (uav.c_t * uav.A_r * uav.rho) + 3 * d**2) * np.sqrt(F * uav.rho * uav.s_r ** 2 * uav.A_r / uav.c_t)
		term2 = (1 + uav.c_f) * F * np.sqrt(np.sqrt((F ** 2 / (4 * uav.rho ** 2 * uav.A_r ** 2)) + (d ** 4 / 4)) - d ** 2 / 2)
		term3 = 0.5 * uav.d_r * d ** 3 * uav.rho * uav.s_r * uav.A_r + uav.m_uav * uav.g * d / uav.n * (np.sin(theta_c))
		P_fly = uav.n * (term1 + term2 + term3)
		
		return P_fly

	def compute_energy_delay(self, loc_uav, loc_ue, offloading_ratio, task_size, real_capacity, uav_fix_distances):
		"""
		计算能耗和时延
		
		Args:
			loc_uav: 无人机位置列表
			loc_ue: 用户位置列表
			offloading_ratio: 卸载比例 [uav_num, ue_num]
			task_size: 任务大小 [ue_num]
			real_capacity: 实际信道容量 [uav_num, ue_num]
			uav_fix_distances: 无人机到固定翼无人机的距离 [uav_num, 1]
		
		Returns:
			t_: 时延数组 [ue_num]
			e_: 能耗数组 [ue_num]
		"""
		# 计算无人机到固定翼无人机的信道增益
		gain_uf = self.beta_0 / (uav_fix_distances ** 2)
		
		r_1 = np.zeros(self.uav_num)
		t1 = np.zeros(self.ue_num)
		e_ = np.zeros(self.ue_num)
		t3 = np.zeros(self.ue_num)
		t_ = np.zeros(self.ue_num)
		
		# 获取无人机参数
		uav0 = self.uavs[0]
		
		# 遍历每个用户
		for j in range(self.ue_num):
			# 找到覆盖该用户的无人机
			covered_uav_idx = -1
			for i in range(self.uav_num):
				if real_capacity[i, j] > 0:
					covered_uav_idx = i
					break
			
			if covered_uav_idx != -1:  # 如果用户被覆盖
				i = covered_uav_idx
				t_upuav = task_size[j] / real_capacity[i, j]  # 用户到无人机的传输时延
				t_comuav = np.float64(task_size[j]) * uav0.C_uav / uav0.f_uav  # 无人机计算时延
				t1[j] = (1 - offloading_ratio[i][j]) * t_comuav  # 无人机计算时延
				
				# 无人机到固定翼无人机的传输速率
				r_1[i] = self.B * math.log(1 + self.uavs[i].P_uav * gain_uf[i] / 10 ** (-13), 2)
				t_uavupfuav = task_size[j] / r_1[i]  # 无人机到固定翼无人机的传输时延
				t_comfuav = np.float64(task_size[j]) * self.fuav.C_fuav / self.fuav.f_fuav  # 固定翼无人机计算时延
				t3[j] = offloading_ratio[i][j] * (t_uavupfuav + t_comfuav)
				t_[j] = t_upuav + max(t1[j], t3[j])
				
				# 计算能耗
				e_[j] = (uav0.K_uav * (uav0.f_uav ** 3) * t1[j] + self.P_ue * t_upuav + 
				         self.uavs[i].P_uav * offloading_ratio[i][j] * t_uavupfuav)
			else:
				# 本地计算
				t_[j] = np.float64(task_size[j]) * self.C_ue / self.f_ue
				e_[j] = 0
		
		return t_, e_

	# ==================== 公平性计算 ====================
	
	def fair(self, cover1, coverage):
		"""
		计算公平性指标
		
		Args:
			cover1: 用户连接情况 [ue_num]
			coverage: 覆盖矩阵 [uav_num, ue_num]
		
		Returns:
			fair_ue: 用户公平性
			fair_uav: 无人机公平性
		"""
		# 用户公平性
		cover2 = np.where(cover1 > 0, 1, 0)
		self.cover_history_ue.append(cover2.tolist())
		cover_history1 = np.array(self.cover_history_ue)
		if len(cover_history1) > 0:
			if np.sum(np.sum(cover_history1, axis=0) ** 2) != 0:
				fair_ue = np.sum(cover_history1) ** 2 / (
						np.sum(np.sum(cover_history1, axis=0) ** 2) * cover_history1.shape[1])
			else:
				fair_ue = 0
		else:
			fair_ue = 0

		# 无人机公平性
		coverage_sum = np.sum(coverage, axis=1)/self.ue_num
		self.cover_history_uav.append(coverage_sum.tolist())
		cover_history2 = np.array(self.cover_history_uav)
		if len(cover_history2) > 0:
			if np.sum(np.sum(cover_history2, axis=0) ** 2) != 0:
				fair_uav = np.sum(cover_history2) ** 2 / (
						np.sum(np.sum(cover_history2, axis=0) ** 2) * cover_history2.shape[1])
			else:
				fair_uav = 0
		else:
			fair_uav = 0

		return fair_ue, fair_uav

	# ==================== 环境主要方法 ====================
	
	def cost(self, loc_uav, loc_ue, offloading_ratio, task_size, record_cover=True):
		"""
		计算成本（时延和能耗）
		
		Args:
			loc_uav: 无人机位置列表
			loc_ue: 用户位置列表
			offloading_ratio: 卸载比例
			task_size: 任务大小
			record_cover: 是否记录覆盖历史
		
		Returns:
			t_: 时延数组
			e_: 能耗数组
			cover1: 用户连接情况
			fair_ue: 用户公平性
			fair_uav: 无人机公平性
		"""
		# 计算距离
		spatial_distances, uav_fix_distances = self.calculate_distances(loc_uav, loc_ue)
		
		# 计算信道容量
		L, capacity = self.capacity(spatial_distances, loc_uav)
		
		# 计算覆盖情况
		coverage, cover1 = self.coverge(spatial_distances, capacity)
		
		# 计算公平性
		fair_ue, fair_uav = self.fair(cover1, coverage)
		
		# 计算实际容量（考虑干扰）
		# real_capacity = self.real_capacity(L, coverage, cover1)
		real_capacity = capacity  # 简化版本，不考虑干扰
		
		# 计算能耗和时延
		t_, e_ = self.compute_energy_delay(loc_uav, loc_ue, offloading_ratio, task_size, 
		                                    real_capacity, uav_fix_distances)
		
		return t_, e_, cover1, fair_ue, fair_uav

	def step(self, actions):
		"""
		环境执行一步
		
		Args:
			actions: 所有无人机的动作列表
		
		Returns:
			state: 新状态
			reward: 奖励
			done: 是否结束
			info: 额外信息
			time: 总时延
			energy_total: 总能耗
		"""
		# 将动作从[-1,1]归一化到[0,1]
		action = [(a + 1) / 2 for a in actions]
		
		uav_energy_list = []
		reward = 0
		
		# 每个无人机执行动作
		for i, uav in enumerate(self.uavs):
			# 解析动作：前3个是移动相关，后面是卸载比例
			move_action = action[i][:3]
			
			# 无人机移动
			success, old_position, d, theta_vertical = uav.move(move_action)
			
			# 计算飞行能耗
			P_fly = self.fly_energy(d, theta_vertical, uav)
			uav_energy_list.append(P_fly)
			
			# 如果超出边界，惩罚
			if not success:
				reward -= 5
		
		# 获取所有无人机的位置
		uav_coords = [uav.get_position() for uav in self.uavs]
		
		# 提取卸载比例
		offload_ratio = [action[i][3:] for i in range(len(action))]
		
		# 计算飞行能耗
		fly_energy = self.w * np.sum(uav_energy_list)
		
		# 计算时延、能耗、覆盖和公平性
		delay, energy, cover, fair_ue, fair_uav = self.cost(uav_coords, self.uecord, 
		                                                      offload_ratio, self.task_list)
		
		# 总能耗
		energy_total = 5 * np.sum(energy) + fly_energy
		time = np.sum(delay)
		
		# 计算奖励
		reward = reward + 20 * fair_uav * fair_ue - (time + energy_total)
		
		# 未覆盖用户的惩罚
		ct = 0
		for x in cover:
			if x == 0:
				ct += 1
		if ct > 6:
			reward -= 1 * ct

		# 无人机之间的安全距离检查
		for i in range(len(self.uavs)):
			for j in range(i + 1, len(self.uavs)):
				distance = np.sqrt(np.sum((self.uavs[i].get_position() - self.uavs[j].get_position()) ** 2))
				safe_distance = 100
				if distance < safe_distance:
					reward -= 5

		# 定义用户簇区域
		user_cluster_1_center = np.array(
			[np.mean([ue[0] for ue in self.cluster1]), np.mean([ue[1] for ue in self.cluster1])])
		user_cluster_2_center = np.array(
			[np.mean([ue[0] for ue in self.cluster2]), np.mean([ue[1] for ue in self.cluster2])])
		user_cluster_radius = 100

		uav_in_cluster = [False, False]

		# 限制无人机在用户分布区域
		for i, uav in enumerate(self.uavs):
			uav_position = np.array([uav.position[0], uav.position[1]])
			distance_to_user_cluster_1 = np.sqrt(np.sum((uav_position - user_cluster_1_center) ** 2))
			distance_to_user_cluster_2 = np.sqrt(np.sum((uav_position - user_cluster_2_center) ** 2))
			if distance_to_user_cluster_1 <= user_cluster_radius:
				uav_in_cluster[i] = 1
				reward += 10
			elif distance_to_user_cluster_2 <= user_cluster_radius:
				uav_in_cluster[i] = 2
				reward += 10

		# 如果两个无人机在同一个用户簇，减少奖励
		if uav_in_cluster[0] == uav_in_cluster[1]:
			reward -= 20

		# 记录信息
		self.reset2(uav_coords, fair_ue, fair_uav, cover, reward, time, energy_total, offload_ratio)
		
		# 更新状态
		self.state = [uav.get_position() for uav in self.uavs]
		
		return self.state, reward, False, {}, time, energy_total

	def reset(self):
		"""重置环境"""
		# 重置公平性历史
		self.cover_history_ue = []
		self.cover_history_uav = []
		
		# 重置无人机位置
		initial_positions = [
			[350, 350, 250],
			[850, 850, 250],
			[350, 950, 250]
		]
		for i, uav in enumerate(self.uavs):
			uav.reset(initial_positions[i])
		
		# 重置用户位置
		self.uecord = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0], [365.74, 971.82, 0],
		                [320.80, 406.66, 0], [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], [267.70, 926.15, 0],
						[966.09, 757.82, 0], [304.61, 786.76, 0], [427.41, 1158.40, 0], [861.97, 925.30, 0], [541.30, 815.78, 0],
		                [413.33, 1023.47, 0], [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], [878.19, 930.27, 0]]

		# 重置任务
		self.task_list = np.random.randint(2021440, 3045729, self.ue_num)
		
		# 更新状态
		self.state = [uav.get_position() for uav in self.uavs]
		
		return self.state

	def reset_step(self):
		"""重置每一步的任务"""
		self.task_list = np.random.randint(2021440, 3045729, self.ue_num)

	def reset2(self, uav_coords, fair_ue, fair_uav, connected_array, reward, time, fly_energy, offratio):
		"""
		记录信息到文件
		
		Args:
			uav_coords: 无人机坐标
			fair_ue: 用户公平性
			fair_uav: 无人机公平性
			connected_array: 连接数组
			reward: 奖励
			time: 时延
			fly_energy: 飞行能耗
			offratio: 卸载比例
		"""
		self.reset_step()
		file_name = '350,850,250(1200x100).txt'
		for i, coords in enumerate(uav_coords):
			with open(file_name, 'a') as file_obj:
				file_obj.write("\nUAV{} hover loc: [{:.2f}, {:.2f}, {:.2f}]".format(i + 1, coords[0], coords[1], coords[2]))
		with open(file_name, 'a') as file_obj:
			file_obj.write("\nFair_ue: {:.2f} Fair_uav: {:.2f}".format(fair_ue, fair_uav))
			file_obj.write("\nConnected Array: " + str(connected_array))
			file_obj.write("\nReward: {:.2f} Time: {:.2f} Energy: {:.2f}".format(reward, time, fly_energy))
		filename = 'off350,850,250(1200x100).txt'
		offratio = [arr.tolist() for arr in offratio]
		with open(filename, 'a') as file_obj:
			file_obj.write("\noffratio" + str(offratio))
