# -*- coding: utf-8 -*-
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from examplebuffer import ReplayBuffer


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma,
                 max_size, C_fc1_dims, C_fc2_dims, C_fc3_dims, A_fc1_dims, A_fc2_dims, batch_size, n_agents,noise,policy_delay=2,per_flag=True):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.noise = noise
        # self.obs_dim = obs_dim
        # self.act_dim = act_dim
        self.replay_buffer = ReplayBuffer(input_dims, n_actions, max_size,)
        self.policy_delay = policy_delay
        # self.noise = OUActionNoise(mu=np.zeros(n_actions*n_agents))
        self.learn_step = 0
        self.per_flag = per_flag
        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='actor')
        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                         n_actions=n_actions, name='target_actor')
        self.critic_1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='critic')
        self.target_critic_1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                n_actions=n_actions, name='target_critic')
        self.critic_2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                      n_actions=n_actions, name='critic')
        self.target_critic_2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, C_fc3_dims, n_agents,
                                             n_actions=n_actions, name='target_critic')
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # print('check out this variable for convergence :', mu)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=self.number_actions ),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, transition):
        self.replay_buffer.store(transition)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self, num_iteration, batch_size1=64):

        # if self.memory.mem_cntr < self.batch_size:
        #     return

        for iteration in range(num_iteration):
            tree_idx, batch_memory, ISWeights = self.replay_buffer.sample(batch_size=batch_size1)
            if isinstance(batch_memory[-1], int):
                continue

            states, actions, rewards, states_, done = [], [], [], [], []

            # states = T.tensor(states, dtype=T.float).to(self.actor.device)
            # states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
            # actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
            # rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
            # done = T.tensor(done).to(self.actor.device)

            # print("batch_memory is an integer:", i,batch_memory[1:-1])

            for idx in range(len(batch_memory)):  # 使用实际长度而非batch_size1
                # 检查数据是否有效
                if not isinstance(batch_memory[idx], (list, tuple)):
                    # 跳过无效数据
                    continue
                
                if len(batch_memory[idx]) < 5:
                    # 数据不完整，跳过
                    continue

                states.append(batch_memory[idx][0])
                actions.append(batch_memory[idx][1])
                rewards.append(batch_memory[idx][2])
                states_.append(batch_memory[idx][3])
                done.append(batch_memory[idx][4])
                # else:
                #     states.append(batch_memory)
                #     actions.append(batch_memory)
                #     rewards.append(batch_memory)
                #     states_.append(batch_memory)
                #     done.append(batch_memory)
            if len(states) == 0:
                print("No valid data in batch_memory, skipping this iteration.")
                continue

            states = T.tensor(np.array(states), dtype=T.float).to(self.actor.device)
            states_ = T.tensor(np.array(states_), dtype=T.float).to(self.actor.device)
            actions = T.tensor(np.array(actions), dtype=T.float).to(self.actor.device)
            rewards = T.tensor(np.array(rewards), dtype=T.float).to(self.actor.device)
            done = T.tensor(np.array(done)).to(self.actor.device)

            self.target_actor.eval()
            self.target_critic_1.eval()
            self.critic_1.eval()
            self.target_critic_2.eval()
            self.critic_2.eval()

            target_actions = self.target_actor.forward(states_)
            critic_value_1_ = self.target_critic_1.forward(states_, target_actions)
            critic_value_1 = self.critic_1.forward(states, actions)
            critic_value_2_ = self.target_critic_2.forward(states_, target_actions)
            critic_value_2 = self.critic_2.forward(states, actions)

            critic_value_1_[done] = 0.0
            critic_value_1_ = critic_value_1_.view(-1)
            critic_value_2_[done] = 0.0
            critic_value_2_ = critic_value_2_.view(-1)

            target = rewards + self.gamma * T.min(critic_value_1_, critic_value_2_)
            target = target.view(-1)  # Flatten the target tensor

            # 标注改动部分
            actual_batch_size = target.size(0)  # 使用实际的批大小
            if actual_batch_size == 0:
                print(f"Warning: empty batch, skipping this iteration.")
                continue  # Skip this batch if empty

            target = target.view(actual_batch_size, 1)
            
            # 确保 critic_value 的维度与 target 一致
            critic_value_1 = critic_value_1.view(actual_batch_size, 1)
            critic_value_2 = critic_value_2.view(actual_batch_size, 1)

            # Optimize Critic 1:
            self.critic_1.train()
            self.critic_1.optimizer.zero_grad()
            critic_loss_1 = F.mse_loss(target, critic_value_1)
            # critic_loss_1.backward(retain_graph=True)
            # self.critic_1.optimizer.step()
            # self.critic_1.eval()

            # Optimize Critic 2:
            self.critic_2.train()
            self.critic_2.optimizer.zero_grad()
            critic_loss_2 = F.mse_loss(target, critic_value_2)
            # critic_loss_2.backward()
            # self.critic_2.optimizer.step()
            # self.critic_2.eval()
            
            # 确保 ISWeights 维度正确
            ISWeights = T.tensor(ISWeights[:actual_batch_size], dtype=T.float32).to(self.actor.device)
            if ISWeights.dim() == 1:
                ISWeights = ISWeights.view(actual_batch_size, 1)
            
            self.abs_errors = T.abs(target - critic_value_1)

            self.q_loss = ISWeights * (critic_loss_1 + critic_loss_2)
            # 将损失张量求和，转换为标量
            self.q_loss = self.q_loss.sum()  # 将损失张量求和，转换为标量
            self.q_loss.backward()  # 对标量损失进行反向传播
            self.critic_1.optimizer.step()
            self.critic_1.eval()
            self.critic_2.optimizer.step()
            self.critic_2.eval()

            if self.learn_step % self.policy_delay == 0:
                self.actor.train()
                self.actor.optimizer.zero_grad()
                actor_loss = -self.critic_1.forward(states, self.actor.forward(states))
                actor_loss = T.mean(actor_loss)
                actor_loss.backward()
                self.actor.optimizer.step()
            self.replay_buffer.batch_update(tree_idx, self.abs_errors.detach().numpy())

            self.learn_step += 1
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params_1 = self.critic_1.named_parameters()
        critic_params_2 = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params_1 = self.target_critic_1.named_parameters()
        target_critic_params_2 = self.target_critic_2.named_parameters()

        critic_state_dict_1 = dict(critic_params_1)
        critic_state_dict_2 = dict(critic_params_2)
        actor_state_dict = dict(actor_params)

        target_critic_state_dict_1 = dict(target_critic_params_1)
        target_critic_state_dict_2 = dict(target_critic_params_2)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict_1:
            critic_state_dict_1[name] = tau * critic_state_dict_1[name].clone() + \
                                      (1 - tau) * target_critic_state_dict_1[name].clone()

        for name in critic_state_dict_2:
            critic_state_dict_2[name] = tau * critic_state_dict_2[name].clone() + \
                                      (1 - tau) * target_critic_state_dict_2[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_state_dict_1)
        self.target_critic_2.load_state_dict(critic_state_dict_2)
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
