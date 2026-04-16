"""
Deep Q-Network (DQN) Agent

Key components:
1. Q-Network: predicts value of actions given state
2. Target Network: lagged copy for stable TD targets
3. Experience Replay: decorrelates training signal
4. Epsilon-Greedy: balances exploration and exploitation

Why DQN for portfolio management?
- Discrete actions map clearly to rebalancing strategies
- Experience replay handles correlated financial time series well
- Target networks prevent Q-value oscillation in nonstationary environments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks import DuelingQNetwork
from models.replay_buffer import ReplayBuffer
from typing import Tuple, Optional


class DQNAgent:
    """
    Deep Q-Network Agent for discrete action spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_frequency: int = 1000,
        device: str = 'cpu'
    ):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            learning_rate: Adam learning rate
            gamma: Discount factor (importance of future rewards)
            epsilon: Initial exploration probability
            epsilon_decay: Decay factor per episode
            epsilon_min: Minimum exploration probability
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_frequency: Steps between target network updates
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device
        
        # Q-networks
        self.q_network = DuelingQNetwork(state_size, action_size).to(device)
        self.target_network = DuelingQNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss (robust to outliers)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training tracking
        self.steps = 0
        self.episode = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (ndarray)
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            action: Integer action index
        """
        # Epsilon-greedy exploration
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Greedy: select best action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Train on a minibatch from replay buffer.
        
        Returns:
            Loss value, or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # === Forward pass ===
        # Current Q-values: Q(s, a)
        q_values = self.q_network(states_t)
        q_values_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # === Target Q-values ===
        # Next Q-values from target network: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t)
            next_q_max = next_q_values.max(dim=1)[0]
            
            # Bellman target: r + gamma * max_a' Q_target(s', a') if not terminal
            targets = rewards_t + self.gamma * next_q_max * (1 - dones_t)
        
        # === Compute loss ===
        loss = self.loss_fn(q_values_taken, targets)
        
        # === Backward pass ===
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.steps += 1
        
        # === Update target network ===
        if self.steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1
    
    def save(self, path: str):
        """Save agent weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Saved DQN agent to {path}")
    
    def load(self, path: str):
        """Load agent weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Loaded DQN agent from {path}")
    
    def get_loss_history(self) -> list:
        """Return training loss history."""
        return self.losses


if __name__ == "__main__":
    # Test DQN agent
    state_size = 106  # 20*5 + 5 + 1
    action_size = 27
    
    agent = DQNAgent(state_size, action_size)
    
    # Dummy experience
    state = np.random.randn(state_size).astype(np.float32)
    action = 5
    reward = 0.5
    next_state = np.random.randn(state_size).astype(np.float32)
    done = False
    
    # Store and train
    for _ in range(50):
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.train_step()
        if loss is not None:
            print(f"Loss: {loss:.6f}")
    
    # Test action selection
    action = agent.select_action(state, training=False)
    print(f"\nSelected action: {action}")
    print(f"Epsilon: {agent.epsilon:.4f}")
