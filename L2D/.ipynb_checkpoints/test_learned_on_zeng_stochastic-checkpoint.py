from mb_agg import *
from agent_utils import *
import torch
import numpy as np
import argparse
from Params import configs
import time
from orliberty import load_instance
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO
import os
import torch
from ortools_scheduler import ORtools_scheduler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

device = configs.device
LOW = configs.low
HIGH = configs.high
benchmark = "zeng"


def test_(times, machines, N_JOBS_N = 20, N_MACHINES_N = 20, max_updates=10000):
    dataset = []
    
    N_JOBS_P,N_MACHINES_P = times.shape
    dataset.append((times,machines))


    env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
            n_j=N_JOBS_P,
            n_m=N_MACHINES_P,
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic)
    path = 'SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH)+'_'+str(max_updates))
    ppo.policy.load_state_dict(torch.load(path))
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                            n_nodes=env.number_of_tasks,
                            device=device)



    result = []
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        while True:
            # Running policy_old:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                graph_pool=g_pool_step,
                                padded_nei=None,
                                adj=adj_tensor,
                                candidate=candidate_tensor.unsqueeze(0),
                                mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward

            if done:
                break
        # print(max(env.end_time))
        # print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)
    t2 = time.time()
    # file_writing_obj = open('./' + 'drltime_' + benchmark + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.txt', 'w')
    # file_writing_obj.write(str((t2 - t1)/len(dataset)))

    return result[0]



def test(instance = "ft06",N_JOBS_N = 20, N_MACHINES_N = 20,max_updates=10000):
    dataset = []
    
    N_JOBS_P,N_MACHINES_P,times,machines = load_instance(instance=instance)
    dataset.append((times,machines))


    env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
            n_j=N_JOBS_P,
            n_m=N_MACHINES_P,
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic)
    path = 'SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH)+'_'+str(max_updates))
    ppo.policy.load_state_dict(torch.load(path))
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                            batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                            n_nodes=env.number_of_tasks,
                            device=device)



    result = []
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        while True:
            # Running policy_old:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                graph_pool=g_pool_step,
                                padded_nei=None,
                                adj=adj_tensor,
                                candidate=candidate_tensor.unsqueeze(0),
                                mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward

            if done:
                break
        # print(max(env.end_time))
        # print('Instance' + str(i + 1) + ' makespan:', -ep_reward + env.posRewards)
        result.append(-ep_reward + env.posRewards)
    t2 = time.time()
    # file_writing_obj = open('./' + 'drltime_' + benchmark + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P) + '.txt', 'w')
    # file_writing_obj.write(str((t2 - t1)/len(dataset)))

    return result[0]
    # print(np.array(result, dtype=np.single).mean())
    # np.save('drlResult_' + benchmark + '_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(N_MACHINES_P), np.array(result, dtype=np.single))


if __name__ == '__main__':
    random_rate = 0.5
    cv = 0.2
    n = 10 
    trained_size = (50, 10)
    max_updates = 10000
    all_instances = ["ft06", "la01", "la06", "la11", "la21", "la31", "la36", "orb01", "yn1", "swv01", "swv06",
                     "swv11", "swv12", "swv13", "swv14", "swv15"]

    
    directory = f"figs_no_future_infomation/p{random_rate}cv{cv}_gin_{str(trained_size)}_{max_updates}_updates"
    if not os.path.exists(directory):
        os.makedirs(directory)

    ortools_mean, ortools_std, policy_mean, policy_std = [], [], [], []
    ortools_on_original, policy_on_original = [], []
    optimal_mean, optimal_std = [], []
    ortools_300s_optimal_rate = []
    current_instances = []

    for instance in all_instances:
        makespan = test(instance=instance, N_JOBS_N=trained_size[0], N_MACHINES_N=trained_size[1],max_updates=max_updates)
        print(instance, makespan)
        sols_directory = f"sols/{instance}/p{random_rate}cv{cv}"

        scheduler = ORtools_scheduler(instance)
        scheduler.read_solution()           # 读取静态解
        obj_val = scheduler.compute_makespan()

        policy_vals, ortools_vals, optimal_vals, if_optimals = [], [], [], []

        for i in range(n):
            scheduler.load_time_mat(os.path.join(sols_directory, f"{i}.npy"))
            times = scheduler.times
            
            # policy_val = scheduler.policy_makespan('dqn', model, shifted_time=times)
            policy_val = test_(times=times, machines=scheduler.machines, N_JOBS_N=trained_size[0], N_MACHINES_N=trained_size[1],max_updates=max_updates)
            ortools_val = scheduler.compute_makespan(shifted_time=times)        # 静态调度面对工时波动

            #if_optimal, optimal_val = scheduler.get_optimal_of_new_time_mat(times)

            policy_vals.append(policy_val)
            ortools_vals.append(ortools_val)
            #optimal_vals.append(optimal_val)
            #if_optimals.append(int(if_optimal))

        info_df = pd.read_csv(os.path.join(sols_directory, "info.csv")) ##############################
        optimal_vals, if_optimals = info_df['obj_val'].values.tolist(), info_df['optimal'].values.tolist()

        plt.plot(policy_vals, color='g', label='policy')
        plt.plot(ortools_vals, color='r', label='ortools_static')
        plt.plot(optimal_vals, color='blue', label='ortools_300s')
        policy_vals, ortools_vals, optimal_vals = np.array(policy_vals), np.array(ortools_vals), np.array(optimal_vals)

        plt.hlines(np.mean(ortools_vals), -2, n+2, linestyles='dotted', colors='r')
        plt.hlines(np.mean(policy_vals), -2, n+2, linestyles='dotted', colors='g')
        plt.hlines(np.mean(optimal_vals), -2, n+2, linestyles='dotted', colors='blue')
        scatter_x = np.where(if_optimals)
        scatter_y = np.array(optimal_vals)[scatter_x]
        plt.scatter(scatter_x, scatter_y, color='blue')



        ortools_mean.append(np.mean(ortools_vals))
        ortools_std.append(np.std(ortools_vals))
        policy_mean.append(np.mean(policy_vals))
        policy_std.append(np.std(policy_vals))
        optimal_mean.append(np.mean(optimal_vals))
        optimal_std.append(np.std(optimal_vals))
        ortools_on_original.append(obj_val)
        policy_on_original.append(makespan)
        ortools_300s_optimal_rate.append(np.mean(if_optimals))
        current_instances.append(instance)

        plt.xlabel('trial')
        plt.ylabel('makespan')
        plt.title(f"random_rate={random_rate},cv={cv},instance={instance}")
        plt.legend()
        plt.savefig(f"{directory}/policy_vs_ortools_{instance}.png")
        plt.clf()

        
    df = pd.DataFrame()
    # 将每个列添加到 DataFrame 中
    df['instance'] = current_instances
    df['ortools_mean'] = ortools_mean
    df['policy_mean'] = policy_mean
    df['optimal_mean'] = optimal_mean
    df['ortools_std'] = ortools_std
    df['policy_std'] = policy_std
    df['optimal_std'] = optimal_std
    df['ortools_on_original'] = ortools_on_original
    df['policy_on_original'] = policy_on_original
    df['ortools_300s_optimal_rate'] = ortools_300s_optimal_rate

    if os.path.exists(f"{directory}/data.csv"):
        df2 = pd.read_csv(f"{directory}/data.csv")
        df2 = df2.append(df, ignore_index=True)
        df2.to_csv(f"{directory}/data.csv", index=False)
    else:
        df.to_csv(f"{directory}/data.csv", index=False)