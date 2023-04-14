import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)




class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        h_dim = 64
        self.feature_extract = GIN(input_dim=2, 
                                   hidden_dim=h_dim, 
                                   n_layers=2).to(device)                                   # input graph nodes' raw features, output 64 dim hidden features
        self.actor_net = mlp([h_dim*2, 32, 32, 1], activation=nn.ReLU).to(device)   # input extracted nodes' features and the selected node's feature
        self.critic = mlp([h_dim, 32, 32, 1], activation=nn.ReLU).to(device)               # input extracted nodes' features
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
    
    def actor(self, states):
        pass



    def take_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        adj, feature, legal_actions = state
        adj, feature = torch.tensor(adj, dtype=torch.float).to(self.device), torch.tensor(feature, dtype=torch.float).to(self.device)
        adj, feature = adj.unsqueeze(0), feature.unsqueeze(0)
        pooled_h, h_nodes = self.feature_extract(adj, feature)
        print(pooled_h.size())
        print(h_nodes.size())
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()





if __name__ == '__main__':
    ##print(net)

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #env_name = 'CartPole-v0'
    #env = gym.make(env_name)
    from GIN_jsspenv import GIN_JsspEnv
    from GIN import GIN
    env_name = 'ft06'
    env = GIN_JsspEnv(env_name)

    env.seed(0)
    torch.manual_seed(0)
    #state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.n
    agent = PPO(actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()