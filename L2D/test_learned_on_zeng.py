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

device = configs.device
LOW = configs.low
HIGH = configs.high
benchmark = "zeng"


def test(instance = "ft06",N_JOBS_N = 20, N_MACHINES_N = 20):
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
    path = 'SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
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



trained_size = (20, 20)
all_instances = ["ft06", "la01", "la06", "la11", "la21", "la31", "la36", "orb01", "yn1", "swv01", "swv06"]
ret = {}
for instance in all_instances:
    result = test(instance=instance, N_JOBS_N=trained_size[0], N_MACHINES_N=trained_size[1])
    print(instance, result)
    ret[instance] = result

