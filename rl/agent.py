from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

from .buffer import ReplayBuffer, PrioritizedReplayBuffer
from .network import Network, Dueling_Network, Dueling_NoisyNetwork


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 1 / 2000,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        noisy = False,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,

        net_type = 'mlp'                                                        # 'mlp', 'gcn'
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha
        )

        # networks: dqn, dqn_target
        self.net_type = net_type
        self.noisy = noisy
        net = Dueling_NoisyNetwork if self.noisy else Dueling_Network
        self.dqn = net(obs_dim, action_dim).to(self.device)
        self.dqn_target = net(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=3e-4)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray, determine: bool=False) -> np.ndarray:
        """Select an action from the input state."""

        # NoisyNet: no epsilon greedy action selection
        if self.noisy or determine:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        else:            # epsilon greedy policy
            if self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(
                    torch.FloatTensor(state).to(self.device)
                ).argmax()
                selected_action = selected_action.detach().cpu().numpy()
        
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()

        # DuelingNet: we clip the gradients to have their norm less than or equal to 10.
        clip_grad_norm_(self.dqn.parameters(), 10.0)

        
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        if self.noisy:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_episodes: int, plot=True, plotting_interval: int = 10, save_plot=None):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        makespans = []

        for episode_idx in range(1, num_episodes + 1):
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                state = next_state
                score += reward
                
                if len(self.memory) >= self.batch_size:             # if training is ready
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()
    


            # PER: increase beta
            fraction = min(episode_idx / num_episodes, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            if not self.noisy:                  # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * fraction
                )
                epsilons.append(self.epsilon)


            if episode_idx % 10 == 0:               # test
                makespans.append(self.test())
                self.is_test = False

            state = self.env.reset()
            scores.append(score)
            score = 0



            # plotting
            if plot:
                if episode_idx % plotting_interval == 0:
                    self._plot(episode_idx, scores, losses, makespans, save_plot=save_plot)

        
        self.env.close()
                
    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state, determine=True)
            next_state, reward, done, info = self.step(action)

            state = next_state
            score += reward


        return info["makespan"]

    
    def _get_dqn(self):
        return self.dqn

    def load_dqn(self, file):
        self.dqn.load_state_dict(torch.load(file))
        self.device = next(self.dqn.parameters()).device

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return loss
    
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        makespans: List[float],
        save_plot=None
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('makespan')
        plt.plot(makespans)
        if save_plot:
            plt.savefig(save_plot)
        plt.show()






class DQNAgent_GCN:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 1 / 2000,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        """
        num_node_features = env.feature_dim
        obs_dim = num_node_features * env.num_nodes
        action_dim = env.action_space.n
        
        self.env = env
        
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha
        )

        # networks: dqn, dqn_target
        net = GCN
        self.dqn = net(num_node_features=num_node_features, num_classes=action_dim).to(self.device)
        self.dqn_target = net(num_node_features=num_node_features, num_classes=action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=3e-4)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray, determine: bool=False) -> np.ndarray:
        """Select an action from the input state."""

        # NoisyNet: no epsilon greedy action selection
        if determine:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device),
                self.env_edge_index
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        else:            # epsilon greedy policy
            if self.epsilon > np.random.random():
                selected_action = self.env.action_space.sample()
            else:
                selected_action = self.dqn(
                    torch.FloatTensor(state).to(self.device),
                    self.env_edge_index
                ).argmax()
                selected_action = selected_action.detach().cpu().numpy()
        
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()

        # DuelingNet: we clip the gradients to have their norm less than or equal to 10.
        clip_grad_norm_(self.dqn.parameters(), 10.0)

        
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, num_episodes: int, plot=True, plotting_interval: int = 10, save_plot=None):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.reset()
        self.env_edge_index = self.env.edge_index
        self.env_edge_index = torch.FloatTensor(self.env_edge_index).to(self.device).long()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        makespans = []

        for episode_idx in range(1, num_episodes + 1):
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)

                state = next_state
                score += reward
                
                if len(self.memory) >= self.batch_size:             # if training is ready
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()
    


            # PER: increase beta
            fraction = min(episode_idx / num_episodes, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * fraction
            )
            epsilons.append(self.epsilon)


            if episode_idx % 10 == 0:               # test
                makespans.append(self.test())
                self.is_test = False

            state = self.env.reset()
            scores.append(score)
            score = 0



            # plotting
            if plot:
                if episode_idx % plotting_interval == 0:
                    self._plot(episode_idx, scores, losses, makespans, save_plot=save_plot)

        
        self.env.close()
                
    def test(self) -> None:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state, determine=True)
            next_state, reward, done, info = self.step(action)

            state = next_state
            score += reward


        return info["makespan"]

    
    def _get_dqn(self):
        return self.dqn

    def load_dqn(self, file):
        self.dqn.load_state_dict(torch.load(file))

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        # print(state.shape)          # [64, 360]
        curr_q_value = self.dqn(state, self.env_edge_index, batch_size=self.batch_size).gather(1, action)
        next_q_value = self.dqn_target(
            next_state, self.env_edge_index, batch_size=self.batch_size
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return loss
    
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        makespans: List[float],
        save_plot=None
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('makespan')
        plt.plot(makespans)
        if save_plot:
            plt.savefig(save_plot)
        plt.show()