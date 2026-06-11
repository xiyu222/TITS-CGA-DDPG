import numpy as np
import torch
from train import *


def evaluate_deviation(agent_idx, nash_agents, sim, num_episodes=50, deviation_samples=20, device=torch.device('cuda')):

    device = device
    num_agents = len(nash_agents)
    baseline_returns = []
    best_deviation_returns = []


    for ep in range(num_episodes):
        state = sim.reset()
        total_reward_baseline = np.zeros(num_agents, dtype=np.float32)
        total_reward_best_dev = np.zeros(num_agents, dtype=np.float32)


        states_seq = []
        other_actions_seq = []
        rewards_seq = []
        done = False


        while True:

            actions = []
            for i in range(num_agents):

                s_i = state[i]
                if isinstance(s_i, np.ndarray):
                    s_tensor = torch.as_tensor(s_i, dtype=torch.float32).to(device)
                else:
                    s_tensor = s_i.to(device).float()

                with torch.no_grad():
                    act_tensor = nash_agents[i].predict_action(s_tensor)
                act_np = act_tensor.cpu().numpy().reshape(-1)
                actions.append(act_np)
            actions_array = np.stack(actions, axis=0)  # shape (num_agents, action_dim)

            next_state, rewards, late, ener = sim.step(actions_array)

            states_seq.append(state)

            other_actions_seq.append(actions_array.copy())
            rewards_seq.append(rewards)
            total_reward_baseline += np.array(rewards, dtype=np.float32)
            state = next_state

            if len(rewards_seq) >= sim.T:
                break

        # baseline done. baseline_returns for this episode:
        baseline_returns.append(total_reward_baseline[agent_idx])


        init_state = states_seq[0]
        best_return = -np.inf

        for ds in range(deviation_samples):

            action_dim = other_actions_seq[0].shape[1]

            dev_act = np.random.uniform(low=0.01, high=0.99, size=(action_dim,))

            state2 = init_state
            total_r = 0.0
            for t in range(sim.T):

                actions2 = []
                for i in range(num_agents):
                    if i == agent_idx:
                        if t == 0:
                            act_i = dev_act
                        else:
                            s_i2 = state2[i]
                            if isinstance(s_i2, np.ndarray):
                                s2_tensor = torch.as_tensor(s_i2, dtype=torch.float32).to(device)
                            else:
                                s2_tensor = s_i2.to(device).float()
                            with torch.no_grad():
                                act_i = nash_agents[i].predict_action(s2_tensor).cpu().numpy().reshape(-1)
                        actions2.append(act_i)
                    else:

                        actions2.append(other_actions_seq[t][i])
                actions2_array = np.stack(actions2, axis=0)
                next_s2, rewards2, late2, ener2 = sim.step(actions2_array)
                total_r += rewards2[agent_idx]
                state2 = next_s2

            if total_r > best_return:
                best_return = total_r
        best_deviation_returns.append(best_return)

    return baseline_returns, best_deviation_returns


def verify_nash_equilibrium(nash_agents, sim, agent_idx_list=None,
                            num_episodes=50, deviation_samples=20, tol=1e-3):

    if agent_idx_list is None:
        agent_idx_list = list(range(len(nash_agents)))
    report = {}
    for idx in agent_idx_list:
        print(f"===  {idx}  ===")
        baseline, best_dev = evaluate_deviation(idx, nash_agents, sim,
                                                num_episodes=num_episodes,
                                                deviation_samples=deviation_samples)
        baseline = np.array(baseline, dtype=np.float32)
        best_dev = np.array(best_dev, dtype=np.float32)
        diffs = best_dev - baseline
        num_better = np.sum(diffs > tol)



    return report

