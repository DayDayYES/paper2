
# -*- coding: utf-8 -*-


import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = list(np.zeros(capacity, dtype=object))  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = transition  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class ReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    这些可以改的，但是目前我也没时间调参了，就凑活用吧
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self,
                 obs_dim,
                 act_dim,
                 size
                 ):
        """
        初始化经验回放缓冲区
        
        Args:
            obs_dim: 观察空间维度
            act_dim: 动作空间维度
            size: 缓冲区大小
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tree = SumTree(size)
        self.full_flag = False
        self.memory_num = 0
        self.memory_size = size

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p
        if self.memory_num < self.memory_size:
            self.memory_num += 1

    def sample(self, batch_size=64):
        n = batch_size
        # n就是batch size！
        # np.empty()这是一个随机初始化的一个矩阵！
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory = []
        
        # 确保 total_p 有效 - total_p 是只读属性，不能直接赋值
        current_total_p = self.tree.total_p
        if current_total_p == 0 or not np.isfinite(current_total_p):
            # 如果 total_p 无效，重置所有优先级为1
            for i in range(self.tree.capacity):
                tree_idx = i + self.tree.capacity - 1
                if self.tree.tree[tree_idx] > 0:  # 只更新有数据的节点
                    self.tree.update(tree_idx, 1.0)
            current_total_p = self.tree.total_p
            if current_total_p == 0:
                current_total_p = 1.0  # 如果还是0，使用默认值
        
        pri_seg = current_total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / current_total_p     # for later calculate ISweight
        if min_prob == 0 or not np.isfinite(min_prob):
            min_prob = 0.00001
        
        valid_count = 0
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            # 确保 a 和 b 是有效的有限值
            if not (np.isfinite(a) and np.isfinite(b)) or a >= b:
                v = current_total_p * (i + 0.5) / n  # 使用中点值
            else:
                v = np.random.uniform(a, b)
            
            idx, p, data = self.tree.get_leaf(v)
            
            # 检查数据是否有效（应该是元组或列表，不是整数）
            if isinstance(data, (list, tuple)) and len(data) >= 5:
                prob = p / current_total_p
                ISWeights[valid_count, 0] = np.power(prob/min_prob, -self.beta)
                b_idx[valid_count] = idx
                b_memory.append(data)
                valid_count += 1
            # 如果数据无效，跳过这个样本
        
        # 如果有效样本不足，返回部分样本
        if valid_count < n:
            b_idx = b_idx[:valid_count]
            ISWeights = ISWeights[:valid_count]
        
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
