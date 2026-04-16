"""
Proximal Policy Optimization (PPO) Agent

Key components:
1. Actor Network: learns policy π(a|s)
2. Critic Network: estimates state value V(s)
3. Advantage Estimation: A(s,a) = r + γV(s') - V(s)
4. Clipped Objective: prevents large policy updates

Why PPO for portfolio management?
- Handles continuous action spaces naturally (continuous allocations)
- More stable than policy gradients (clipped objective)
- Better sample efficiency than DQN on continuous control
- Easily parallelizable (we use serial version here for clarity)

PPO Key Ideas:
1. Collect trajectory data with current policy
2. Compute advantages using value function
3. Update policy to maximize clipped advantage
4. Update value function to match returns
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.networks import ActorNetwork, CriticNetwork
from typing import Tuple, Optional


class PPOAgent:
    """
    Proximal Policy Optimization Agent for continuous action spaces.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Args:
            state_size: State dimension
            action_size: Action dimension
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_ratio: Clip ratio for policy updates
            epochs: Number of epochs per update
            batch_size: Batch size for training
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
            value_loss_coef: Coefficient for value function loss
            device: 'cpu' or 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size).to(device)
        self.critic = CriticNetwork(state_size).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        self.policy_losses = []
        self.value_losses = []
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Select action from policy.
        
        Args:
            state: Current state
            training: If True, sample; if False, use deterministic policy
            
        Returns:
            action: Continuous action in [0, 1]
            log_prob: Log probability of action (for training)
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get action and log probability
            action_t, log_prob_t = self.actor.get_action_and_log_prob(state_t)
            action = action_t.squeeze(0).cpu().numpy()
            log_prob = log_prob_t.item()
        
        return action.astype(np.float32), log_prob
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in trajectory buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def estimate_value(self, state: np.ndarray) -> float:
        """
        Estimate state value using critic network.
        
        Args:
            state: Current state
            
        Returns:
            Estimated value
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_t).squeeze(0).item()
        return value
    
    def compute_advantages_and_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        GAE provides a good tradeoff between bias and variance in advantage estimates.
        
        Args:
            None (uses stored trajectory)
            
        Returns:
            advantages: Advantage estimates
            returns: Target returns (for value function training)
        """
        advantages = []
        returns = []
        gae = 0
        
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute GAE backwards through trajectory
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0  # Bootstrap value at end
            else:
                next_value = values[t + 1]
            
            # TD residual
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE accumulation
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantage = gae
            ret = advantage + values[t]
            
            advantages.insert(0, advantage)
            returns.insert(0, ret)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages (improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """
        Update policy and value function using trajectory data.
        
        This is the core PPO update:
        1. Compute advantages
        2. Update policy using clipped objective
        3. Update value function
        """
        if len(self.states) == 0:
            return None, None
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns()
        
        # Convert to tensors
        states_t = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # === PPO updates ===
        policy_loss_sum = 0
        value_loss_sum = 0
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(len(self.states))
            
            for start_idx in range(0, len(self.states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                states_batch = states_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                old_log_probs_batch = old_log_probs_t[batch_indices]
                advantages_batch = advantages_t[batch_indices]
                returns_batch = returns_t[batch_indices]
                
                # === Policy update ===
                # Get current policy's action probability
                action_batch, log_probs_batch = self.actor.get_action_and_log_prob(
                    states_batch, actions_batch
                )
                
                # Probability ratio: π_new / π_old
                prob_ratio = torch.exp(log_probs_batch - old_log_probs_batch)
                
                # Clipped surrogate objective (PPO's key contribution)
                surr1 = prob_ratio * advantages_batch
                surr2 = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus (encourages exploration)
                dist = torch.distributions.Normal(action_batch, torch.exp(self.actor.log_std))
                entropy = dist.entropy().mean()
                
                policy_loss = policy_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                
                policy_loss_sum += policy_loss.item()
                
                # === Value update ===
                values_batch = self.critic(states_batch).squeeze(-1)
                value_loss = nn.MSELoss()(values_batch, returns_batch)
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                
                value_loss_sum += value_loss.item()
                self.steps += 1
        
        # Store average losses
        num_updates = self.epochs * (len(self.states) // self.batch_size)
        avg_policy_loss = policy_loss_sum / max(num_updates, 1)
        avg_value_loss = value_loss_sum / max(num_updates, 1)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        
        # Clear trajectory buffer
        self.clear_trajectory()
        self.episodes += 1
        
        return avg_policy_loss, avg_value_loss
    
    def clear_trajectory(self):
        """Clear trajectory buffer after update."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save(self, path: str):
        """Save agent weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Saved PPO agent to {path}")
    
    def load(self, path: str):
        """Load agent weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Loaded PPO agent from {path}")


if __name__ == "__main__":
    # Test PPO agent
    state_size = 106
    action_size = 5
    
    agent = PPOAgent(state_size, action_size)
    
    # Dummy experience
    state = np.random.randn(state_size).astype(np.float32)
    action, log_prob = agent.select_action(state)
    
    reward = 0.5
    value = agent.estimate_value(state)
    done = False
    
    # Store multiple transitions
    for i in range(20):
        agent.store_transition(state, action, reward, value, log_prob, done)
    
    # Update
    policy_loss, value_loss = agent.update()
    print(f"Policy loss: {policy_loss:.6f}")
    print(f"Value loss: {value_loss:.6f}")
