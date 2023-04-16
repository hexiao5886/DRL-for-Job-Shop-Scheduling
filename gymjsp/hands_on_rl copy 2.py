import gym
import time
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

class Actor(nn.Module):
    def __init__(self, feature_extract, h_dim, device):
        super(Actor, self).__init__()
        self.feature_extract =  feature_extract
        self.mlp = mlp([h_dim*2, 32, 32, 1], activation=nn.ReLU).to(device)
        self.device = device
        
    
    def forward(self, states):
        batched_probs = []

        for state in states:
            adj, feature, mask, candidate_operation_indexes = state
            adj, feature, mask = torch.tensor(adj, dtype=torch.float).to(self.device), \
                torch.tensor(feature, dtype=torch.float).to(self.device), torch.tensor(mask, dtype=torch.float).to(self.device)
            adj, feature, mask = adj.unsqueeze(0), feature.unsqueeze(0), mask.unsqueeze(0)
            pooled_h, h_nodes = self.feature_extract(adj, feature)
            # print(pooled_h.size())                                    # [1, 64]
            # print(h_nodes.size())                                     # [1, 36, 64]
            # print(candidate_operation_indexes)                         # [0, 6, 12, 18, 24, 30]
            h_candiates = h_nodes[:,candidate_operation_indexes,:]      
            # print(h_candiates.size())                                   # # [1, 6, 64]
            pooled_h_expanded = pooled_h.expand_as(h_candiates)
            # print(pooled_h_expanded.size())                             # [1, 6, 64]
            concateFea = torch.cat((pooled_h_expanded, h_candiates), dim=-1)
            # print(concateFea.size())                                    # [1, 6, 128]
            candidate_scores = self.mlp(concateFea).squeeze(-1)
            # print(candidate_scores.size())                              # [1, 6]
            # perform mask
            candidate_scores[mask==1] = float('-inf')
            probs = F.softmax(candidate_scores, dim=1)
            batched_probs.append(probs)

        batched_probs = torch.cat(batched_probs, dim=0)

        return batched_probs


    
class Critic(nn.Module):
    def __init__(self, feature_extract, h_dim, device):
        super(Critic, self).__init__()
        self.feature_extract =  feature_extract
        self.mlp = mlp([h_dim, 32, 32, 1], activation=nn.ReLU).to(device)
        self.device = device
        
    def forward(self, states):
        batched_pooled_h = []

        for state in states:
            adj, feature, mask, candidate_operation_indexes = state
            adj, feature, mask = torch.tensor(adj, dtype=torch.float).to(self.device), \
                torch.tensor(feature, dtype=torch.float).to(self.device), torch.tensor(mask, dtype=torch.float).to(self.device)
            adj, feature, mask = adj.unsqueeze(0), feature.unsqueeze(0), mask.unsqueeze(0)
            pooled_h, h_nodes = self.feature_extract(adj, feature)

            batched_pooled_h.append(pooled_h)

        batched_pooled_h = torch.cat(batched_pooled_h, dim=0)       # pooled_h is [1,64] and batched_pooled_h is [batch, 64]

        v = self.mlp(batched_pooled_h)

        return v


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        h_dim = 64
        self.feature_extract = GIN(input_dim=2, 
                                   hidden_dim=h_dim, 
                                   n_layers=2).to(device)           # input graph nodes' raw features, output 64 dim hidden features
        
        # actor and critic share feature extract
        self.actor = Actor(feature_extract=self.feature_extract,
                           h_dim=h_dim, device=device)
        self.critic = Critic(feature_extract=self.feature_extract,
                             h_dim=h_dim, device=device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device



    def take_action(self, state):
        probs = self.actor([state])
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        # extract (s, a, r, s_, done) from transition_dict
        states = transition_dict['states']
        next_states = transition_dict['next_states']
        
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # Critic only input the whole graph feature
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # print(self.performe_GIN_extract(states).size())                       # [36, 64]


        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self):
        time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        torch.save(self.actor.state_dict(), time_stamp + "_actor.pth")
        torch.save(self.critic.state_dict(), time_stamp + "_critic.pth")

    def load(self, actor_file, critic_file):
        self.actor.load_state_dict(torch.load(actor_file))
        self.critic.load_state_dict(torch.load(critic_file))

if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 10
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    from GIN_jsspenv import GIN_JsspEnv
    from GIN import GIN
    env_name = 'ft06'
    env = GIN_JsspEnv(env_name)
    env.seed(0)
    torch.manual_seed(0)
    agent = PPO(actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list, makespan_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

    episodes_list = list(range(len(return_list)))


    mv_return = rl_utils.moving_average(return_list, 9)
    mv_makespan = rl_utils.moving_average(makespan_list, 9)
    plt.plot(episodes_list, mv_return, label='return')
    plt.plot(episodes_list, mv_makespan, label='makespan')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.legend()
    import time
    
    plt.savefig(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+'.png')
    plt.show()