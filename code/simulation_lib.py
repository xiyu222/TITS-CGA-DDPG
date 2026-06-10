import numpy as np
from collections import namedtuple
import random
import copy
import torch

seed = 512
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
device = torch.device('cuda')


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
State = namedtuple('State', ('t', 'p', 'q'))


class State(State):
    def getNormalizedState(self, toTensor=True):
        norm_q = self.q / 200
        norm_p = (self.p - 110) / 100
        norm_t = (self.t - 12) / 12
        out = copy.deepcopy(np.concatenate((np.array([norm_t, norm_p]), norm_q)))

        if toTensor:
            return out
        else:
            return torch.from_numpy(out)

    def getState(self):
        return copy.deepcopy(np.concatenate((np.array([self.t, self.p]), self.q)))


def get_tran_speed(band_rate, all_W):
    delta_2 = 1
    tran_speed = all_W * band_rate * np.log2(1 + (10 * 3.5) / (1+2))
    return tran_speed


class Device:
    def __init__(self, C):
        self.weight_energy = 10  # 能量消耗权重
        self.weight_delay = 5  # 延迟权重
        self.weight_sinr = 60  # 信噪比权重
        self.C = C * 100 * 40  # 任务量, bit
        self.loc_cal_rate = 3 * 1000 * 1000  # 本地处理速度 1Mbit/s
        self.edge_cal_rate = 300 * 1000 * 1000  # 云端处理速度 200Mbit/s
        self.loc_cal_energy_coe = 20  # 本地能量消耗系数
        self.p = 5  # 传输功率
        self.downlink_rate= 100 # 下行传输功率
        self.loc_queue_rate = 10    # 本地每秒处理 10 个任务的能力
        self.edge_queue_rate = 30   # 边缘每秒处理 30 个任务的能力

class MarketSimulator(object):
    def __init__(self, param_dict, user_num, his_len, C=1000):
        self.p_imp = param_dict['price_impact']
        self.t_cost = param_dict['transaction_cost']
        self.L_cost = param_dict['liquidation_cost']
        self.phi = param_dict['running_penalty']
        self.T = param_dict['T']
        self.dt = param_dict['dt']
        self.N = param_dict['N_agents']
        self.user_num = user_num
        self.his_len = his_len
        self.mu = param_dict['drift_function']
        self.sigma = param_dict['volatility']
        self.sigma0 = param_dict['initial_price_var']
        self.device = Device(C)
        self.C = C

        self.r = lambda Q, S, nu: - nu * (S + self.t_cost * nu) - self.phi * Q ** 2

        self.Q = np.random.normal(0, self.sigma0, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.dS = np.float32(0)
        self.dF = np.float32(0)
        self.t = np.float32(0)

        self.state_map = torch.from_numpy(np.zeros(
            (self.user_num, (self.user_num + 1) * self.his_len), dtype='float32'))
        self.Sa = torch.from_numpy(np.zeros((self.user_num, self.user_num + 1), dtype='float32'))

        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))

    def reset(self):
        self.Q = np.random.normal(0, 10, self.N)
        self.S = np.float32(10 + np.random.normal(0, self.sigma))
        self.t = np.float32(0)

        self.last_reward = np.zeros(self.N, dtype=np.float32)
        self.total_reward = np.zeros(self.N, dtype=np.float32)

        self.dW = np.random.normal(0, np.sqrt(self.dt),
                                   int(round(np.ceil(self.T / self.dt) + 2)))

        for i in range(self.user_num):
            flag = 0
            action = np.random.random(self.user_num - 1)
            bandwidth = np.random.random(self.user_num)
            computing_resource = np.random.random(self.user_num)

            for j in range(self.user_num):
                if j != i:
                    self.Sa[i, flag] = action[flag]
                    flag += 1
            if flag != self.user_num - 1:
                print("Index error!")
            else:
                self.Sa[i, flag] = bandwidth[flag]
                self.Sa[i, flag + 1] = computing_resource[flag]

        return self.Sa

    def calculate_energy(self, x, sum_band, sum_com):
        ener = []
        for i in range(self.user_num):
            # 上传（上行）能耗: P_up * t_up
            tran_energy = 0.1 * x[i, 0] * (self.device.C / 1000) * self.device.p

            # 下载（下行）能耗: P_dl * t_dl
            # 假设下行传输功率比上行小 1/4，且使用相同的数据量和带宽分配
            dl_power = self.device.p * 0.01
            dl_energy = dl_power * x[i, 0] * (self.device.C / 1000) / (self.device.downlink_rate * x[i, 1] * sum_band)

            # 本地计算能耗: 设备本地计算功耗 * 计算时长
            loc_cal_energy = 0.04 * (
                        1 - x[i, 0]) * self.device.C / 1000 / self.device.loc_cal_rate * self.device.loc_cal_energy_coe

            # 边缘计算能耗: 边缘服务器计算功耗 * 计算时长
            edg_cal_energy = 0.01 * x[i, 0] * (self.device.C) / 1000 / (
                        self.device.edge_cal_rate * x[i, 2] * sum_com) * self.device.loc_cal_energy_coe

            # print(tran_energy)
            # print(dl_energy)

            # 总能耗: 上行 + 下行 + 本地计算 + 边缘计算
            total_energy = tran_energy + dl_energy + loc_cal_energy + edg_cal_energy
            ener.append(total_energy)
        return ener

    def step(self, x):
        sum_off = 0.0
        sum_band = 0.0
        sum_com = 0.0

        for i in range(self.user_num):
            sum_off += x[i, 0] * self.device.C

        for i in range(self.user_num):
            sum_band += x[i, -3]

        for i in range(self.user_num):
            sum_com += x[i, -2]

        m_time = np.array([0, 0, 0])
        sum_reward = 0
        latency = 0
        energy = 0

        ener = []
        late = []
        re = []



        # 更新用户状态
        for i in range(self.user_num):
            # 计算能耗
            ener = self.calculate_energy(x, sum_band, sum_com)

            # 上行传输速率与延迟
            up_speed = get_tran_speed(x[i, 1], sum_band * 1000)
            tran_latency = 0.05 * x[i, 0] * self.device.C / up_speed

            # 本地与边缘排队等待延迟（指数分布模拟）
            queue_loc = np.random.exponential(1.0 / self.device.loc_queue_rate)
            queue_edge = np.random.exponential(1.0 / self.device.edge_queue_rate)

            # 本地计算延迟 + 本地队列等待
            loc_compute = 0.05 * (1 - x[i, 0]) * self.device.C / self.device.loc_cal_rate
            loc_cal_latency = loc_compute + queue_loc

            # 边缘计算延迟 + 边缘队列等待
            edge_compute = 0.05 * x[i, 0] * self.device.C
            edge_compute /= (self.device.edge_cal_rate * (x[i, 2] * sum_com))
            edge_cal_latency = edge_compute + queue_edge

            # 下行传输延迟（简化为与上行速率相同）
            dl_speed = up_speed
            dl_latency = 0.05 * x[i, 0] * self.device.C / dl_speed

            # 计算本地计算与上行传输的关键路径
            uplink_path = max(loc_cal_latency, tran_latency)

            # 汇总：并行执行+边缘计算+下行传输+微小抖动
            cal_latency = uplink_path + edge_cal_latency + dl_latency
            cal_latency += 0.0005 * np.random.random()

            # 最终延迟增加整体系统抖动
            total_latency = cal_latency + 15 * np.random.random()
            late.append(total_latency)


            # 计算奖励
            # sum_reward = -1 * (0.5 * cal_latency + 0.5 * ener[i])

            T_max =15
            alpha =2
            # 计算超出的时延（若未超时，则为 0）
            exceed_time = max(0.0, cal_latency - T_max)

            # 计算线性惩罚
            penalty = alpha * exceed_time

            # 原有的 reward 加上超时惩罚
            sum_reward = -1 * (0.5 * cal_latency + 0.5 * ener[i] + penalty)

            re.append(sum_reward)

        for i in range(self.user_num):
            flag = 0
            for j in range(self.user_num):
                if j != i:
                    self.Sa[i, flag] = x[j, 0]
                    flag += 1
            if flag != self.user_num - 1:
                print("Index error!")
            else:
                self.Sa[i, flag] = x[i, -2]
                self.Sa[i, flag + 1] = x[i, -1]

        return self.Sa, re, late, ener