import numpy as np
import torch


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from simulation_lib import *
from nashRL_netlib import *
from NashAgent_lib import *
from matplotlib import pyplot as plt
# from  ppo import train_model

seed = 512
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
device = torch.device('cuda')

# -------------------------------------------------------------------
# 该文件执行 Nash-DQN强化学习算法
# -------------------------------------------------------------------

# 设置 numpy 的打印精度为 4 位小数
np.set_printoptions(precision=4)

# 定义训练和模型参数
user_num = 3 # 总智能体数
T = 15  # 总时间步数

# 定义仿真参数
sim_dict = {'price_impact': .3,
            'transaction_cost': .5,
            'liquidation_cost': .5,
            'running_penalty': 0,
            'T': T,
            'dt': 1,
            'N_agents': user_num,
            'drift_function': (lambda x, y: 0.1 * (10 - y)),  # x -> time, y-> price
            'volatility': 1,
            'initial_price_var': 20}


# 模拟市场环境
sim = MarketSimulator(sim_dict, user_num=user_num, his_len=1)


def warm_up(run_time,nash_agent):
    for k in range(0,run_time):
        current_state = sim.reset()
        rewards = [list() for i in range(user_num)]
        action = np.zeros((user_num, 3))

        # 每个智能体独立选择
        for i in range(user_num):
            action[i] = nash_agent[i].predict_action(current_state[i]).data.cpu().numpy()


        next_state, reward, late, ener = sim.step(action)
        for i in range(user_num):
            rewards[i].append(reward[i])
            m_action = torch.tensor(action[i], dtype=torch.float32)
            m_reward = torch.tensor(reward[i], dtype=torch.float32)
            exp = nash_agent[i].experiencePool.experience(current_state[i], m_action, m_reward, next_state[i])
            nash_agent[i].experiencePool.save_experience(exp)
            nash_agent[i].warm_up()
        # print(action)

# 主函数用于运行 Nash 算法，参数包括最大训练轮数、最大时间步数、批量更新大小、缓冲区大小等。
def run_Nash_Agent(MAX_EPISOIDE=1500, MAX_TIME_STEP=15,batch_update_size=100, buffersize=5000, AN_file_name="Action_Net",
                   VN_file_name="Value_Net"):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """

    # 需要网络估计的参数个数
    max_action = 100  # 可采取最大行动的大小

    # Set number of output variables needed from net:
    # (c1 + c2 + c3 + mu)
    parameter_number = 4

    # 创建仿真环境

    #Estimate/actual transaction costs (used to improve convergence of nash value)
    est_tr_cost = sim_dict['transaction_cost']

    # #Estimated Liquidation cost (of selling/buying shares past last timestep)
    term_cost = sim_dict['liquidation_cost']

    # 初始化 NashNN Agents
    '''
    parameter_number:
        1.卸载率
        2.带宽分配
        3.计算资源分配

    '''
    nash_agent = []

    for i in range(user_num):
        nash_agent.append(NashNN(4, 3, user_num, T, est_tr_cost, term_cost, num_moms=5))
        # nash_agent.append(train_model(PermInvariantQNNActor()))
        # nash_agent = NashNN(2+num_players,parameter_number,num_players,T,est_tr_cost,term_cost,num_moms = 5)
    rewards = [list() for i in range(user_num)]
    latency = []
    energy = []
    reward_list = []
    # 应该为每一个用户设置一个网络
    # user_num = 3
    # user_drl = []
    # for i in range(user_num):
    #     user_drl.append(NashNN())

    # current_state = sim.get_state()[0]

    # exploration chance
    ep = 0.5  # Initial chance
    min_ep = 0.1  # Minimum chance

    # 初始化终极内存
    # replay = ExperienceReplay(buffersize)

    # sum_loss = np.zeros(num_sim)
    # total_l = 0


    # 初始化动作
    action = np.zeros((user_num, 3))
    reward_list = []
    # Set feasibility exploration space of inventory levels:
    # space = np.array([-100,100])

    # ---------- Main simulation Block -----------------
    train_flag = 1
    # current_state = sim.reset()
    warm_up(10,nash_agent)

    # 在每一轮训练中，智能体根据当前状态选择动作，与环境交互获得奖励和下一个状态
    for k in range(0, MAX_EPISOIDE):

        train_flag += 1
        # # Decays Exploration rate Linearly and Resets Loss
        # eps = max (max( ep - (ep-0.05)*(k/(num_sim-1)), 0 ),min_ep)
        # total_l = 0
        #
        # # Sets Print Flag - Prints simulation results every 20 simuluations


        # for timestep in range(0,MAX_TIME_STEP):
        # # 初始化环境

        current_state = sim.reset()
        #
        #     if np.random.random() < eps: #这里是探索率，如果需要智能体随即探索，可以启用这一步
        #         #Set target level of inventory level to cover feasible exploration space
        #         # then select action so it results in that inventory level
        #         target_q = np.random.multivariate_normal(np.ones(num_players)*(space[1]+space[0])/2,np.diag(np.ones(num_players)*(space[1]-space[0])/4))
        #         a = target_q - current_state.p
        #     else:
        #         test = nash_agent.predict_action([current_state])[0].mu
        #         a = nash_agent.predict_action([current_state])[0].mu.cpu().data.numpy()

        for i in range(user_num):

            # 每个智能体独立选择
            # action[i] = 0.05 #edge-only
            action[i] = nash_agent[i].predict_action(current_state[i]).data.cpu().numpy()
            #     if k > 5:
            #         action[i] = nash_agent[i].predict_action(current_state[i]).data.cpu().numpy()
            #     else :
            #         action[i] = np.random.random(user_num)
        # a = nash_agent.predict_action([current_state])[0].mu.cpu().data.numpy()
        # a = trunc_array(a, 1, 0)

        # Take Chosen Actions and Take Step
        # b = a.data
        next_state, reward, late, ener = sim.step(action)
        np_late = np.array(late)
        all_late = np.median(np_late)
        np_ener = np.array(ener)
        all_ener = np.median(np_ener)
        # all_ener = 0

        latency.append(all_late)
        energy.append(all_ener)
        # print("reward : {}, latency : {}, energy : {}".format(reward.item(), late.item(), ener.item()))
        for i in range(user_num):
            rewards[i].append(reward[i])
            m_action = torch.tensor(action[i], dtype=torch.float32)
            m_reward = torch.tensor(reward[i], dtype=torch.float32)

            # 训练

            nash_agent[i].store((current_state[i], m_action ,m_reward,next_state[i]))
            if nash_agent[i].sumtree.data.poolsize > 100 and train_flag % 1 == 0:
                nash_agent[i].train(k)
        if k % 100 == 0:
            print('Episode:',k)
            print(action)   
        reward_list.append(rewards[0])



    #数据输出、制图
    np.savetxt(r'X:\代码数据\学习率重复\Reward\0.0009-999.reward' + str(user_num) + '.txt',
               reward_list[MAX_EPISOIDE - 1])
    np.savetxt(r'X:\代码数据\学习率重复\Energy\0.0009-999.energy' + str(user_num) + '.txt', energy)
    np.savetxt(r'X:\代码数据\学习率重复\Latency\0.0009-999.latency' + str(user_num) + '.txt', latency)


    #生成图像
    plt.figure()
    plt.plot(reward_list[MAX_EPISOIDE-1])
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(r'X:\代码数据\学习率重复\Reward\0.0009-999_reward.png')
    plt.close()

    plt.figure()
    plt.plot(energy)
    plt.title('Energy Consumption')
    plt.xlabel('Episode')
    plt.ylabel('Energy')
    plt.savefig(r'X:\代码数据\学习率重复\Energy\0.0009-999_energy.png')
    plt.close()

    plt.figure()
    plt.plot(latency)
    plt.title('Processing Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.savefig(r'X:\代码数据\学习率重复\Latency\0.0009-999_latency.png')
    plt.close()

    # reward_save = np.array(rewards[0])
    # plt.plot(reward_save)
    # plt.show()
    # plt.plot(reward_list[MAX_EPISOIDE-1])
    # plt.show()
    plt.plot(reward_list[MAX_EPISOIDE-1])
    plt.show()
    plt.plot(energy)
    plt.show()
    plt.plot(latency)
    plt.show()


