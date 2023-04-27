import os
import torch
from rl.agent import DQNAgent
from gymjsp.jsspenv import HeuristicJsspEnv
from ortools_scheduler import ORtools_scheduler
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from copy import deepcopy

num_episodes = 1000
memory_size = 100000
batch_size = 64
target_update = 100
noisy = False
plotting_inteval = 10

instances = ["ft06", "la01", "la06", "la11", "la21", "la31", "la36", "orb01", "swv01", "swv06", "swv11", "yn1"]
# num_episodes of swv11 is actually 100, changed to 1000 for implementation simplicity

def load_agent(instance):
    policy_file = f"policies/dqn_mlp/{instance}_num_episodes={num_episodes}_memory_size={memory_size}_target_update={target_update}_noisy={noisy}.pth"
    env = HeuristicJsspEnv(instance)
    agent = DQNAgent(env, memory_size, batch_size, target_update, noisy=noisy)
    agent.load_dqn(policy_file)
    return agent

def get_makespan_of_random_policy(env, num_simulations=10):
    makespans = []
    for _ in range(num_simulations):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 8)
            next_state, reward, done, info = env.step(action)
            state = next_state

        makespan = info["makespan"]
        makespans.append(makespan)
    return np.mean(makespans)

def get_makespan_of_heuristic_rule(env, rule):
    """rule can be between 0 and 1"""
    state = env.reset()
    done = False
    while not done:
        next_state, reward, done, info = env.step(rule)
        state = next_state
    return info["makespan"]

def get_makespan_of_agent_policy(env, agent):
    state = env.reset()
    done = False
    agent_actions = []
    while not done:
        action = agent.select_action(state, determine=True)
        agent_actions.append(int(action))
        next_state, reward, done, info = env.step(action)
        state = next_state
    return info["makespan"]