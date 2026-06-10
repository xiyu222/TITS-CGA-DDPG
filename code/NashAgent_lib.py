import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nashRL_netlib import *
from collections import namedtuple
import random
import torch.nn.functional as F
device = torch.device('cuda')

#tf.compat.v1.disable_eager_execution()

seed = 512
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.

# 基于强化学习的纳什均衡策略训练框架


# 这个函数用于生成一个组合形状的元组，通常用于初始化数组或张量时指定形状
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# 简单的经验回放池，用于存储和采样智能体的经验
class randomExperiencePool:
    # 初始化经验池
    def __init__(self, size):
        self.experience = namedtuple("agent_experience", ['state', 'action', 'reward', 'next_state'])
        self.experience_pool = []
        self.poolSize = size
        self.capcity = 0

    # 方法将新的经验添加到池中，当池满时移除最早的经验
    def save_experience(self, exp):
        if len(self.experience_pool) >= self.poolSize:
            self.experience_pool = self.experience_pool[1:]  # 保持经验池经验个数恒定
            self.capcity = len(self.experience_pool)
        self.experience_pool.append(exp)
        self.capcity += 1

    # 方法从池中随机采样一批经验，返回状态、动作、奖励和下一个状态
    def sample(self, batch_size=64):
        batch_experience = random.choices(self.experience_pool, k=batch_size)
        batch_experience = self.experience(*zip(*batch_experience))
        state = torch.cat(batch_experience.state)
        action = torch.cat(batch_experience.action)
        reward = torch.cat(batch_experience.reward)
        done = torch.cat(batch_experience.done)
        return state, action, reward, done

# SumTree 是一种数据结构
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    # 初始化 SumTree，设置了容量和数据结构
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = PreExperenicePool(obs_dim,act_dim,capacity)
        # [--------------data frame-------------]
        #             size: capacity

    # 添加新的经验并更新优先级
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data.store(*data)
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0               # replay buffer也会同步将指针指向初始位置

    # 更新指定节点的优先级
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 方法根据优先级采样叶子节点
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
        return leaf_idx, self.tree[leaf_idx], self.data.sample_one(data_idx)

    # 属性返回根节点的总优先级
    @property
    def total_p(self):
        return self.tree[0]  # the root


# 预经验池 存储和采样经验数据
class PreExperenicePool(object):
    # 初始化状态、动作、奖励和下一个状态的缓冲区
    def __init__(self,s_dim,a_dim,size):
        self.state_buf = np.zeros(combined_shape(size,s_dim),dtype=np.float64)
        self.next_state_buf = np.zeros(combined_shape(size,s_dim),dtype=np.float64)
        self.action_buf = np.zeros(combined_shape(size,3),dtype=np.float64)
        # self.reward_buf = np.zeros(combined_shape(size,s_dim),dtype=np.float64)
        self.reward_buf = np.zeros(size,dtype=np.float64)
        # self.action_buf = np.zeros(size,dtype=np.float64)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.poolsize = size

    #  方法存储单个经验
    def store(self,state,action,reward,next_state):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.ptr = (self.ptr +1 ) % self.max_size
        self.size = min(self.size+1,self.max_size)


    # 采样一批经验
    def sample_batch(self,batch_size=100):
        idxs = np.random.randint(0,self.size,batch_size)
        batch = dict(state = self.state_buf[idxs],
                     next_state = self.next_state_buf[idxs],
                     action = self.action_buf[idxs],
                     reward = self.reward_buf[idxs],
                     )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    # 采样单个经验
    def sample_one(self,index):
        # index = random.randint(0,index)
        batch = dict(state = self.state_buf[index],
                     next_state = self.next_state_buf[index],
                     action = self.action_buf[index],
                     reward = self.reward_buf[index],
                     )
        return batch


# 定义纳什神经网络
class NashNN():
    """
    Object summarizing estimated parameters of the advantage function, initiated
    through a vector of inputs
    :param input_dim:    Number of total input features
    :param output_dim:   Number of total parameters to be estimated via NN
    :param nump:         Number of agents
    :param t:            Number of total time steps
    :param t_cost:       Transaction costs (estimated or otherwise)
    :param term_cost:    Terminal costs (estimated or otherwise)
    """

    # 方法初始化网络、优化器、损失函数以及经验池和 SumTree
    def __init__(self, input_dim, output_dim, nump, t, t_cost, term_cost, num_moms = 5):
        self.num_players = nump
        # self.T = t
        # self.transaction_cost = t_cost
        # self.term_costs = term_cost
        self.state_dim = nump + 1

        self.action_net = PermInvariantQNNActor(in_invar_dim = input_dim, non_invar_dim = 3, out_dim = output_dim, num_moments=num_moms)
        self.value_net = PermInvariantQNNCritic(in_invar_dim = input_dim + output_dim, non_invar_dim = 3, out_dim = 1)
        self.value_net_target = PermInvariantQNNCritic(in_invar_dim = input_dim + output_dim, non_invar_dim = 3, out_dim = 1)

        # Define optimizer used (SGD, etc)
        # self.optimizer_DQN = optim.RMSprop(list(self.action_net.moment_encoder_net.parameters()),
        #
        #                                lr=0.007)
        # self.optimizer_value = optim.RMSprop(list(self.value_net.moment_encoder_net.parameters()),
        #                                lr=0.007)
        self.optimizer_DQN = optim.Adam(list(self.action_net.moment_encoder_net.parameters()),
                                       lr=0.0005)
        self.optimizer_value = optim.Adam(list(self.value_net.moment_encoder_net.parameters()),
                                       lr=0.0005)

        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        # self.experiencePool = Memory(capacity=5000)
        self.experiencePool = randomExperiencePool(5000)
        self.action_dim = nump + 1
        self.sumtree = SumTree(5000,nump+1,3)
        self.abs_error_upper = 10
        self.gamma = 0.001
        self.beat = 0.999


    #  计算Q值损失
    def comput_loss_q(self,data,leaf_idx_list,ISweight):
        # discount = 0.001
        state = data['state']
        action = data['action']
        reward = data['reward']
        next_state = data['next_state']
        # cur_Q = self.value_net(state, action)
        # critic_loss = F.mse_loss(cur_Q, target_Q.view(1,-1))
        q = self.value_net(state,action)
        with torch.no_grad():
            next_action = self.action_net(next_state)
            next_action = (next_action).clamp(0.01, 0.99)
            target_Q = self.value_net_target(next_state, next_action)
            reward = reward.to(device)
            backup = reward + self.gamma * target_Q * 1000 * 30
            # print(target_Q)
            # print(reward)
            # print(backup)
            abs_td_error = torch.abs(q-backup.view(1,-1))
            for i,leaf_idx in enumerate(leaf_idx_list):
                p = abs_td_error[i] + 0.01
                p = torch.clip(p,0,self.abs_error_upper)
                p = torch.pow(p,1)
                self.sumtree.update(leaf_idx,p[i].item())

            temp = (q-backup)**2
            temp = temp.to(device)
            ISweight = ISweight.to(device)
            loss_q = (torch.mul(temp.view(-1,100), ISweight)).mean()

        return loss_q


    # 定义计算动作损失的方法
    def compute_loss_action(self,data):
        state = data['state']
        action = self.action_net(state)
        action_loss = self.value_net(state,action)
        return -action_loss.mean()


    # 定义训练方法
    # 根据 SumTree 的总优先级，将优先级范围划分为多个段。
    # 从每个段中随机采样一个值，获取对应的经验。
    # 计算重要性采样权重。
    # 将数据转换为 PyTorch 张量。
    # 使用 comput_loss_q 计算 Q 值的损失，并更新价值网络的参数。
    # 使用 compute_loss_action 计算动作的损失，并更新策略网络的参数。
    # 软更新目标网络的参数。
    # 返回 Q 值的损失。
    def train(self, episode ,batch_size = 100):
        #将train网络和target网络合并时的权重参数
        tau = 0.5
        total = self.sumtree.total_p
        seg = total / batch_size
        data = {'state':[],'action':[],'reward':[],'next_state':[]}
        leaf_idx_list = list()
        self.beat = np.min((self.beat+0.001,1))
        ISweight = np.empty(batch_size)
        for i in range(batch_size):
            begin, end = seg * i, seg*(i+1)
            v = np.random.uniform(begin,end)
            tree_idx, p, batch = self.sumtree.get_leaf(v)
            leaf_idx_list.append(tree_idx)
            data['state'].append(batch['state'])
            data['action'].append(batch['action'])
            # for i in range(len(batch['reward'])):
            data['reward'].append(batch['reward'])
            data['next_state'].append(batch['next_state'])
            # data['done'].append(batch['done'])
            ISweight[i] = np.power(p/total,-self.beat)
        ISweight = torch.as_tensor(ISweight / np.max(ISweight),dtype=torch.float64)
        data = {k: torch.as_tensor(np.array(v), dtype=torch.float32) for k, v in data.items()}

        self.optimizer_value.zero_grad()
        loss_q = self.comput_loss_q(data,leaf_idx_list,ISweight)
        loss_q.requires_grad_(True)
        loss_q.backward()
        self.optimizer_value.step()

        # self.optimizer_DQN.zero_grad()
        action_loss = self.compute_loss_action(data)
        self.optimizer_DQN.zero_grad()
        action_loss.backward()
        self.optimizer_DQN.step()


        for train_parameters,target_parameters in zip(self.value_net.parameters(), self.value_net_target.parameters()):
            target_parameters.data.copy_(tau * train_parameters.data + (1 - tau) * target_parameters.data)

        return loss_q


    #  定义存储经验的方法
    def store(self,transimision):
        max_p = np.max(self.sumtree.tree[-5000:])
        if max_p == 0:
            max_p = self.abs_error_upper
        self.sumtree.add(max_p,transimision)

    # 定义预测动作的方法
    def predict_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        # expanded_states = torch.tensor(self.expand_list(states)).float()
        # action_list = self.action_net.forward(invar_input = expanded_states[:,3:].cuda(), non_invar_input = expanded_states[:,0:3].cuda())
        # 对action_list进行归一化
        states = states.view(-1, self.state_dim)
        action = self.action_net(states)
        action = torch.clamp(action, 0.01, 0.99)
        NFV_list = []
        # for i in range(0,len(states)):
        #     NFV_list.append(NashFittedValues(action_list[i*self.num_players:(i+1)*self.num_players,:]))
        #
        # return NFV_list

        return action




    # 这段代码实现了一个DDPG智能体的预热训练过程，通过采样经验池中的数据更新价值网络和策略网络，并使用软更新策略保持目标网络的稳定性。
    def warm_up(self):
        tau = 0.5
        batch_experience = random.choices(self.experiencePool.experience_pool, k=100)  # 对经验池进行批处理
        batch_experience = self.experiencePool.experience(*zip(*batch_experience))
        state = torch.cat(batch_experience.state)
        action = torch.cat(batch_experience.action)
        reward = []
        for i in range(len(batch_experience.reward)):
            reward.append(batch_experience.reward[i].item())
        # reward = torch.cat(batch_experience.reward)
        # done = torch.cat(batch_experience.done)
        next_state = torch.cat(batch_experience.next_state)
        reward = torch.FloatTensor(reward)

        # 开始训练
        with torch.no_grad():
            next_action = self.action_net(next_state)
            next_action = (next_action).clamp(0.01, 0.99)
            target_Q = self.value_net_target(next_state, next_action)
            reward = reward.to(device)
            backup = reward + self.gamma * target_Q * 1000 * 30
        cur_Q = self.value_net(state, action)
        loss_q = F.mse_loss(cur_Q, backup)
        self.optimizer_value.zero_grad()
        loss_q.backward()
        self.optimizer_value.step()

        action = self.action_net(state)
        action_loss = -self.value_net(state, action).mean()
        self.optimizer_DQN.zero_grad()
        action_loss.backward()
        self.optimizer_DQN.step()
        # # # # print(action)
        # 更新参数
        for train_parameters, target_parameters in zip(self.value_net.parameters(), self.value_net_target.parameters()):
            target_parameters.data.copy_(tau * train_parameters.data + (1 - tau) * target_parameters.data)

