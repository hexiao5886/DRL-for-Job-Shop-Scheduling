import os
import random
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from rl.agent import DQNAgent_GCN
from gymjsp.jsspenv import HeuristicJsspEnv,HeuristicGraphJsspEnv




def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



def train_dqn_gcn(instance, num_episodes = 100, memory_size = 100000, batch_size = 64, target_update = 50,
              plotting_interval=10, seed=777, save_plot=None):
    env = HeuristicGraphJsspEnv(instance)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)


    agent = DQNAgent_GCN(env, memory_size, batch_size, target_update)
    agent.train(num_episodes, plotting_interval=plotting_interval, save_plot=save_plot)

    trained_dqn = agent._get_dqn()
    makespan = agent.test()

    return trained_dqn, makespan


