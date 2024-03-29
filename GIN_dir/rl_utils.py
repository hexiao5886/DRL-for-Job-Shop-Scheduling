from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from collections import defaultdict
from collections import OrderedDict
from GIN_jsspenv import GIN_JsspEnv
from logger import Logger

        




def train_on_policy_agent(env, agent, num_episodes, logger, logs):
    
    for itr in range(num_episodes):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        agent.eval()
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        actor_loss, critic_loss = agent.update(transition_dict)

        logs["makespan"] = info["makespan"]
        logs["return"] = episode_return
        logs["actor_loss"] = actor_loss
        logs["critic_loss"] = critic_loss

        # perform the logging
        for key, value in logs.items():
            #print('{} : {}'.format(key, value))
            logger.log_scalar(value, key, itr)
        
        logger.flush()



def train_on_policy_agent_parallel_envs(instances, agent, num_iterations, num_episodes_per_iter, return_lists, makespan_lists):
    """
    envs: list of environments
    agent: agent
    num_iterations: number of iterations
    num_episodes_per_iter: number of episodes per iteration
    return_lists: list of lists of returns
    makespan_lists: list of lists of makespans
    """
    for i in range(num_iterations):
        n = len(instances[0])
        idx = i % n
        idx_instances = [instance[idx] for instance in instances]
        for _ in range(num_episodes_per_iter):
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            for instance in idx_instances:
                time_mat, machine_mat = instance
                env = GIN_JsspEnv(processing_time_matrix=time_mat, machine_matrix=machine_mat)
                env.seed(0)
                episode_return = 0
                state = env.reset()
                done = False
                agent.eval()
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                makespan_lists[env.name].append(info["makespan"])
                return_lists[env.name].append(episode_return)
            actor_loss, critic_loss = agent.update(transition_dict)               # update agent after collecting data from envs




def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)