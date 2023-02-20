import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Batch
import warnings
warnings.filterwarnings('ignore')


from gymjsp.jsspenv import HeuristicJsspEnv
from tianshou.env import SubprocVectorEnv
from gymjsp.orliberty import load_random, load_instance

def tianshou_ppo_train(instance_name, max_epoch=1, schedule_cycle=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def helper_function():
        env = HeuristicJsspEnv(instance_name, schedule_cycle=schedule_cycle)
        return env

    env = HeuristicJsspEnv(instance_name, schedule_cycle=schedule_cycle)
    train_envs = SubprocVectorEnv([helper_function for _ in range(100)])
    test_envs = SubprocVectorEnv([helper_function for _ in range(10)])


    # net is the shared head of the actor and the critic
    net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
    actor = Actor(net, env.action_space.n, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    # optimizer of the actor and the critic
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    dist = torch.distributions.Categorical
    policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(50000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=max_epoch,
        step_per_epoch=50000,
        repeat_per_collect=5,
        episode_per_test=1,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda mean_reward: mean_reward >= 195,
    )

    # Let's watch its performance!
    policy.eval()

    obs = env.reset(random=False)
    done = False
    score = 0

    while not done:
        action = policy(Batch(obs=obs[np.newaxis, :])).act.item()
        next_state, reward, done, info = env.step(action)

        obs = next_state
        score += reward

    return info["makespan"], policy