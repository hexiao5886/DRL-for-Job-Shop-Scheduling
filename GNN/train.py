import torch
import numpy as np
import matplotlib.pyplot as plt
from GIN_jsspenv import GIN_JsspEnv
from agent import PPO
import warnings
warnings.filterwarnings('ignore')
import time
from collections import defaultdict
from collections import OrderedDict
from GIN_jsspenv import GIN_JsspEnv
from logger import Logger


def train_on_policy_agent(env, agent, num_episodes, logger, step):
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

        logs = OrderedDict()
        logs["makespan"] = info["makespan"]
        logs["return"] = episode_return
        logs["actor_loss"] = actor_loss
        logs["critic_loss"] = critic_loss

        # perform the logging
        for key, value in logs.items():
            #print('{} : {}'.format(key, value))
            logger.log_scalar(value, key, itr + step)
        
        # logger.flush()



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





def train_single_size_on(size, num_episodes_per_env=10, dir="logdir/"):
    env_name = f"train_single_size_on_{size}"
    exp_name = f"num_episodes_per_env={num_episodes_per_env}"
    logdir = dir + exp_name + '_' + env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logger = Logger(log_dir=logdir)
    

    num_jobs, num_machines = size
    generated_instances_file = f"DataGen/generatedData{num_jobs}_{num_machines}_Seed200.npy"
    policy_dir = f"saved_policies/train_single_size_on_{num_jobs}_{num_machines}_num_episodes_per_env={num_episodes_per_env}"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    actor_lr = 2e-5
    critic_lr = 2e-5
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    h_dim = 64
    
    logger.log_scalar(actor_lr, "actor_lr", 0)
    logger.log_scalar(critic_lr, "critic_lr", 0)
    logger.log_scalar(gamma, "gamma", 0)
    logger.log_scalar(lmbda, "lmbda", 0)
    logger.log_scalar(epochs, "epochs", 0)
    logger.log_scalar(eps, "eps", 0)
    logger.log_scalar(h_dim, "h_dim", 0)


    agent = PPO(device=device, actor_lr=actor_lr, critic_lr=critic_lr, lmbda=lmbda, epochs=epochs,eps=eps,gamma=gamma,h_dim=h_dim)
    generated_instances = np.load(generated_instances_file)

    for i in range(len(generated_instances)):
        instance = generated_instances[i]
        time_mat, machine_mat = instance
        env = GIN_JsspEnv(processing_time_matrix=time_mat, machine_matrix=machine_mat)
        env.seed(0)
        train_on_policy_agent(env=env, agent=agent, num_episodes=num_episodes_per_env, logger=logger, step=i*num_episodes_per_env)

    agent.save(policy_dir)


if __name__ == '__main__':
    # sizes = [(6,6),(10,10),(15,10),(15,15),(20,10),(20,15),(20,20),(30,15),(30,20),(50,20),(100,20),(200,50)]
    
    train_single_size_on(size=(6,6), num_episodes_per_env=1)