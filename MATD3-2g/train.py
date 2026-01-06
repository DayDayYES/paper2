import random
import matplotlib.pyplot as plt
import numpy as np
from uav_env import UAVTASKENV
import os
import scipy.io
from perddpg_torch import Agent
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from buffer import ReplayBuffer
from scipy.interpolate import make_interp_spline
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 或者任何你想要的级别
# training of the agents in the environment using MADDPG


# function to concatenate states of each agent for the critic
def obs_list_to_state_vector(observation):
	state = np.array([])
	for obs in observation:
		state = np.concatenate([state, obs])
	return state


# creating two UE devices cluster on 600X600 sqmeter area
def create_UE_cluster(x1, y1, x2, y2):
	X = []
	Y = []
	Z = []
	while (len(X) < 10):
		cord_x = round(random.uniform(x1, x2), 2)
		if (cord_x not in X):
			X.append(cord_x)
	while (len(Y) < 10):
		cord_y = round(random.uniform(y1, y2), 2)
		if (cord_y not in Y):
			Y.append(cord_y)
	while (len(Z) < 10):
		Z.append(0)
	k = []
	i = 0
	while i < 10:
		k.append([X[i], Y[i], Z[i]])
		i += 1

	return k

# ue_cluster_1 = create_UE_cluster(400, 450, 470, 520)  # 矩形区域内随机生成10个用户的坐标
# ue_cluster_2 = create_UE_cluster(30, 30, 100, 100)

# ue_cluster_1 = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0],[565.74, 471.82, 0],
#                 [320.80, 406.66, 0], [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], [667.70, 426.15, 0]]
# ue_cluster_2 = [[966.09, 757.82, 0], [1004.61, 986.76, 0], [927.41, 1158.40, 0], [861.97, 925.30, 0], [1041.30, 1015.78, 0],
#                 [1013.33, 1023.47, 0], [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], [878.19, 930.27, 0]]

# ue_cluster_1 = [[391.03, 433.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0], [565.74, 471.82, 0],
# 				[320.80, 406.66, 0]]
# ue_cluster_2 = [[1004.61, 986.76, 0], [927.41, 1158.40, 0],
# 				[749.50, 965.05, 0], [784.61, 685.57, 0], [878.19, 930.27, 0]]

ue_cluster_1 = [[391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], [352.51, 636.99, 0],[365.74, 971.82, 0],
                [320.80, 406.66, 0], [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], [267.70, 926.15, 0]]
ue_cluster_2 = [[966.09, 757.82, 0], [304.61, 786.76, 0], [427.41, 1158.40, 0], [861.97, 925.30, 0], [541.30, 815.78, 0],
                [413.33, 1023.47, 0], [749.50, 965.05, 0], [784.61, 685.57, 0], [899.15, 852.64, 0], [878.19, 930.27, 0]]



# main loop
if __name__ == '__main__':
	batch_size = 64
	memory_size = 1000000
	gamma = 0.99  # discount factor
	alpha = 0.0001  # learning rate
	beta = 0.001
	update_actor_interval = 2
	noise = 0.2
	# actor and critic hidden layers
	C_fc1_dims = 512
	C_fc2_dims = 256
	C_fc3_dims = 256

	A_fc1_dims = 1024
	A_fc2_dims = 512

	x_axis_data = []
	tau = 0.005

	env = UAVTASKENV(ue_cluster_1, ue_cluster_2)
	n_agents = env.uav_num  # 智能体数量
	actor_dims = []  # 每个智能体的观察空间维度，无人机的三维坐标
	for i in range(n_agents):
		actor_dims.append(3)
	critic_dims = sum(actor_dims)
	PRINT_INTERVAL = 1

	n_actions = 3+len(ue_cluster_1+ue_cluster_2)  # 动作空间维度  无人机的距离和垂直和水平方向
	agents = []
	for index_agent in range(n_agents):
		agent = Agent(alpha, beta, 3, tau, n_actions, gamma, memory_size, C_fc1_dims, C_fc2_dims, C_fc3_dims,
		              A_fc1_dims, A_fc2_dims, batch_size, n_agents, noise)

		agents.append(agent)

	memory = ReplayBuffer(100000, 3, n_actions)

	total_steps = 0
	score_history = []
	n_episodes = 800
	timestamp = 100
	avg = []
	train_rewards = []
	a_bound = [-1, 1]

	# 记录开始时间
	start_time = time.time()
	start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
	print("Start time: ", start_time_formatted)

	# standard implemntaion of MADDPG algorithim
	for i in range(n_episodes):
		obs = env.reset()
		# obs_tensors = [tf.convert_to_tensor(o, dtype=tf.float32) for o in obs]
		# obs = tf.convert_to_tensor(obs, dtype=tf.float32)
		score = 0
		episode_step = 0
		t = []
		energy = []

		for j in range(timestamp):
			action_all = []
			for agent_idx in range(n_agents):
				action = agents[agent_idx].choose_action(obs[agent_idx])
				action = np.clip(action, *a_bound)
				action_all.append(action)

			obs_, reward, done, info, t_, f_energy = env.step(action_all)

			state = obs_list_to_state_vector(obs)
			state_ = obs_list_to_state_vector(obs_)

			# memory.store_transition(obs, action_all, reward, obs_, done)

			# taking the agents actions, states and reward
			for agent_idx in range(n_agents):
				agents[agent_idx].remember((obs[agent_idx], action_all[agent_idx], reward, obs_[agent_idx], done))

			# agents take random samples and learn
			for agent_idx in range(n_agents):
				agents[agent_idx].learn(2)

			# if total_steps % 100 == 0:  # learn every 10 steps
			# 	maddpg_agents.learn(memory)
			# if memory.mem_cntr > 1000000:
			# 	# var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
			# 	maddpg_agents.learn(memory)
			obs = obs_

			score += reward
			total_steps += 1
			episode_step += 1

			t.append(t_)
			energy.append(f_energy)

			# if j == slot_num - 1:
			# 	print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % score)

			# file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'

		score_history.append(score)  # store the episodic reward
		train_rewards.append(score)
		avg_score = np.mean(score_history[-100:])  # average reward over the last 100 episodes
		avg_score = avg_score / timestamp
		t = np.mean(t)
		energy = np.mean(energy)
		if i % PRINT_INTERVAL == 0 and i > 0:
			print('episode', i, 'average score {:.2f}'.format(avg_score), 'reward {:.2f}'.format(score), 'time {:.2f}'.format(t), 'energy {:.2f}'.format(energy))
			avg.append(avg_score)
			file_name = '350,850,250(1200x100).txt'
			with open(file_name, 'a') as file_obj:
				file_obj.write("\n========================= This episode is done =========================")  # 本episode结束
				file_obj.write("\nAverage score: " + str(avg_score))

	# 记录结束时间
	end_time = time.time()
	end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
	print("End time: ", end_time_formatted)

	# 计算并显示总运行时间
	total_time = end_time - start_time
	print(f"Total training time: {total_time:.2f} seconds")

	# 获取当前工作目录
	current_directory = os.getcwd()

	# 构建保存路径
	save_path_re = os.path.join(current_directory, 'Reward850-350-250(1200x100).png')
	save_path_av = os.path.join(current_directory, 'Avg850-350-250(1200x100).png')

	# plot the final results
	# maddpg_agents.save_checkpoint()
	plt.plot(avg)
	plt.xlabel("Episode")
	plt.ylabel("Avg. Epsiodic Reward")
	plt.savefig(save_path_av)
	plt.show()

	plt.figure()
	# plot the training rewards
	plt.plot(train_rewards)
	plt.xlabel("Episode")
	plt.ylabel("Epsiodic Reward")
	plt.savefig(save_path_re)
	plt.show()

