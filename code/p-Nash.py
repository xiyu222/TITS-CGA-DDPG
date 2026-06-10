import numpy as np
import torch
from train import *
# 假设：sim 是 MarketSimulator 实例；nash_agents 是训练好的 NashNN 实例列表
# 需要根据你的脚本结构，确保这里能访问到 sim 和 nash_agents。

def evaluate_deviation(agent_idx, nash_agents, sim, num_episodes=50, deviation_samples=20, device=torch.device('cuda')):
    """
    对单个智能体 agent_idx，在多个初始情形下评估基线策略与若干偏离策略的收益差。
    - nash_agents: 训练好的智能体列表
    - sim: MarketSimulator 实例
    - num_episodes: 用多少次独立环境重置采样不同初始过程
    - deviation_samples: 每次episode中，针对初始状态或每一步，对 agent_idx 生成多少偏离动作样本尝试
    返回:
      baseline_returns: list of length num_episodes，使用原策略得到的累计回报
      best_deviation_returns: list of length num_episodes，使用最优偏离动作策略得到的累计回报
    近似评估如果 best_deviation_returns[i] > baseline_returns[i] 明显较多，则认为存在可改进偏离。
    """
    device = device
    num_agents = len(nash_agents)
    baseline_returns = []
    best_deviation_returns = []

    # 偏离动作生成方式：简单随机采样动作向量；若想更精细可用局部扰动或小范围梯度搜索等。
    for ep in range(num_episodes):
        state = sim.reset()  # 假设 reset() 返回每个智能体所需状态，可以是 list/ndarray
        # 记录环境中各智能体累计回报
        total_reward_baseline = np.zeros(num_agents, dtype=np.float32)
        total_reward_best_dev = np.zeros(num_agents, dtype=np.float32)

        # 首先，用训练好的策略执行整个episode，记录轨迹中各步骤状态、动作、奖励，以便后续偏离测试
        # 这里假设环境长度固定 T 步，或者 sim.step 返回 done 信号也可处理
        # 我们先记录所有（state_sequence, other_agents_actions, rewards）等
        states_seq = []
        other_actions_seq = []  # 记录其他智能体的动作，便于后续固定
        rewards_seq = []        # 记录每一步的即时奖励
        done = False

        # 先运行一次“基线策略”轨迹，记录数据
        # 假设 sim.step 接受一个 shape=(num_agents, action_dim) 的动作数组，返回 (next_state, rewards, late, ener)
        # 这里只关注 rewards；如果需要多步累积，也可以考虑折扣奖励等
        # 注意：如果环境含随机成分，最好固定随机种子或多次采样评估。这里简单取一次
        while True:
            # 构造各智能体动作
            actions = []
            for i in range(num_agents):
                # 拼装 state 类型：NashNN.predict_action 期望 torch tensor on device
                # 假设 state[i] 是 numpy array 或 torch tensor
                s_i = state[i]
                if isinstance(s_i, np.ndarray):
                    s_tensor = torch.as_tensor(s_i, dtype=torch.float32).to(device)
                else:
                    s_tensor = s_i.to(device).float()
                # Add batch dim if需要: predict_action 中做了 view(-1, self.state_dim)
                # 假设 predict_action 接受形如 torch tensor of shape [state_dim] or [1, state_dim]
                with torch.no_grad():
                    act_tensor = nash_agents[i].predict_action(s_tensor)
                act_np = act_tensor.cpu().numpy().reshape(-1)  # 一维动作向量
                actions.append(act_np)
            actions_array = np.stack(actions, axis=0)  # shape (num_agents, action_dim)
            # 执行动作
            next_state, rewards, late, ener = sim.step(actions_array)
            # 记录
            states_seq.append(state)
            # 记录其他智能体动作，便于后续偏离时固定
            other_actions_seq.append(actions_array.copy())
            rewards_seq.append(rewards)
            total_reward_baseline += np.array(rewards, dtype=np.float32)
            state = next_state
            # 如果环境有 done 信号，可在此判断；否则假设固定步数 T，sim internally handles.
            # 这里假设 sim.reset()/step 会自行在固定步数后停止或需要外部计数：
            # 如果 sim.step 没有 done，且环境长度固定，你可在外部用 for t in range(T): 来循环
            # 此处简化为：若 sim 设计为固定 T，可以改写为 for t in range(T): ...
            # 假设 sim.step 永远返回有效，且我们用一个隐藏计数限制最大步数:
            # TODO: 如果需要，请自行在此加入 for t in range(T) 结构。这里示例仅做思路。
            # 假设环境固定 T 步，则可 break when len(rewards_seq) == T:
            if len(rewards_seq) >= sim.T:  # 假设 sim 有属性 T
                break

        # baseline done. baseline_returns for this episode:
        baseline_returns.append(total_reward_baseline[agent_idx])

        # 接下来，对 agent_idx 做“偏离”尝试：在同一初始状态序列下，固定其他智能体动作，替换 agent_idx 的动作，重新运行环境，看看累计回报
        # 注意：为了更严谨，应当从每一步开始尝试偏离，但这里简化：仅在初始状态时偏离并沿该偏离动作执行，或者在每一步都随机采样偏离动作并模拟序列。
        # 下面示例：仅在第 0 步偏离（更多细化可在每一步做循环）。
        # 初始状态：
        init_state = states_seq[0]
        best_return = -np.inf
        # 生成若干偏离动作样本
        for ds in range(deviation_samples):
            # 随机生成一个偏离动作：可在动作空间内随机采样
            # 假设动作维度为 action_dim，且动作范围 [0.01,0.99]（与训练中 clamp 一致）
            action_dim = other_actions_seq[0].shape[1]
            # 简单随机均匀采样：
            dev_act = np.random.uniform(low=0.01, high=0.99, size=(action_dim,))
            # 重新运行一个 episode：从 init_state 开始，第一步用 dev_act，之后每步 agent_idx 也用同样 dev_act 或继续用基线策略？
            # 这里示例：偏离仅在第一步使用 dev_act，之后沿用基线策略。也可设计更复杂：后续每步都随机偏离或固定偏离策略。
            state2 = init_state
            total_r = 0.0
            for t in range(sim.T):
                # 构造动作列表：对 agent_idx 用 dev_act（t==0），其后可继续用基线或同一 dev_act
                actions2 = []
                for i in range(num_agents):
                    if i == agent_idx:
                        if t == 0:
                            act_i = dev_act
                        else:
                            # 这里示例：后续沿用基线策略
                            s_i2 = state2[i]
                            if isinstance(s_i2, np.ndarray):
                                s2_tensor = torch.as_tensor(s_i2, dtype=torch.float32).to(device)
                            else:
                                s2_tensor = s_i2.to(device).float()
                            with torch.no_grad():
                                act_i = nash_agents[i].predict_action(s2_tensor).cpu().numpy().reshape(-1)
                        actions2.append(act_i)
                    else:
                        # 其他智能体固定使用原来 baseline 轨迹中的动作（也可以每步重新用策略生成，保持一致性）
                        # 这里用 baseline 轨迹中记录的 other_actions_seq[t][i]
                        actions2.append(other_actions_seq[t][i])
                actions2_array = np.stack(actions2, axis=0)
                next_s2, rewards2, late2, ener2 = sim.step(actions2_array)
                total_r += rewards2[agent_idx]
                state2 = next_s2
            # 比较
            if total_r > best_return:
                best_return = total_r
        best_deviation_returns.append(best_return)

    return baseline_returns, best_deviation_returns


def verify_nash_equilibrium(nash_agents, sim, agent_idx_list=None,
                            num_episodes=50, deviation_samples=20, tol=1e-3):
    """
    针对 nash_agents 列表中的每个智能体，评估基线策略与最优偏离策略的收益。
    - agent_idx_list: 如果只想验证部分智能体，可传入索引列表；否则对所有智能体都验证。
    - num_episodes, deviation_samples: 评估时的采样数量，可根据计算资源调节。
    - tol: 如果 best_deviation_return - baseline_return > tol，即认为存在改进偏离。
    最终打印或返回一个结果报告：对于每个智能体，在多少次评估中发现可改进偏离。
    """
    if agent_idx_list is None:
        agent_idx_list = list(range(len(nash_agents)))
    report = {}
    for idx in agent_idx_list:
        print(f"=== 验证智能体 {idx} 的纳什平衡情况 ===")
        baseline, best_dev = evaluate_deviation(idx, nash_agents, sim,
                                                num_episodes=num_episodes,
                                                deviation_samples=deviation_samples)
        baseline = np.array(baseline, dtype=np.float32)
        best_dev = np.array(best_dev, dtype=np.float32)
        diffs = best_dev - baseline
        # 统计在哪些 episode 中发现偏离更好
        num_better = np.sum(diffs > tol)
        print(f"智能体 {idx}: 共 {num_episodes} 次评估，发现更好偏离的次数: {num_better}")
        print(f"平均基线收益: {baseline.mean():.4f}, 平均最优偏离收益: {best_dev.mean():.4f}, 差值均值: {diffs.mean():.4f}")
        if num_better == 0:
            print(f"→ 对于智能体 {idx}，在给定评估设置下，未发现能明显改进的偏离，近似满足纳什均衡条件。\n")
        else:
            print(f"→ 对于智能体 {idx}，存在 {num_better}/{num_episodes} 次可改进偏离，可能未到达纳什均衡，建议进一步训练或更大范围搜索。\n")
        report[idx] = {
            'baseline_mean': float(baseline.mean()),
            'best_dev_mean': float(best_dev.mean()),
            'num_better': int(num_better),
            'diffs': diffs.tolist()
        }
    return report
if __name__ == '__main__':
    # 训练
    nash_agents = run_Nash_Agent(MAX_EPISOIDE=1500, MAX_TIME_STEP=15, batch_update_size=100,
                                 buffersize=5000, AN_file_name="Action_Net", VN_file_name="Value_Net")
    # 假设 sim 已在全局或以适当方式创建
    # 如果 sim 在 run_Nash_Agent 内部创建，需要在外部重新创建同样参数的 sim：
    # sim = MarketSimulator(sim_dict, user_num=user_num, his_len=1)
    # 或者将 sim 设为全局变量，训练结束后依旧可访问。

    # 验证纳什均衡
    report = verify_nash_equilibrium(nash_agents, sim,
                                     num_episodes=30,    # 可先用较小次数测试
                                     deviation_samples=10, tol=1e-3)
    # report 包含每个智能体的评估结果，可根据需要保存或打印更详细信息
    # 例如：
    import json
    with open('nash_equilibrium_report.json','w') as f:
        json.dump(report, f, indent=2)
    print("验证完成，报告已保存到 nash_equilibrium_report.json")
