"""
Neural Network Architectures

Q-Network (DQN): Dueling architecture for stable value estimates
Policy Network (PPO): Actor-Critic for continuous control
Value Network (PPO): Estimates state value

These are optimized for the portfolio problem:
- Moderate state size (~100 dimensions)
- Discrete or continuous actions
- Financial domain (stable, continuous optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network.
    
    Architecture:
        Shared layers → (Value stream + Advantage stream)
        Q(s,a) = V(s) + (A(s,a) - mean(A))
    
    Why dueling? 
    - Separates value (how good is this state) from advantage (how good is this action)
    - More stable learning because we don't need to estimate both equally
    - Better convergence, especially when most actions are similar
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        dueling: bool = True
    ):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.dueling = dueling
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        if dueling:
            # Value stream: estimates V(s)
            self.fc_value = nn.Linear(hidden_size, hidden_size // 2)
            self.value = nn.Linear(hidden_size // 2, 1)
            
            # Advantage stream: estimates A(s,a) for each action
            self.fc_advantage = nn.Linear(hidden_size, hidden_size // 2)
            self.advantage = nn.Linear(hidden_size // 2, action_size)
        else:
            # Standard DQN: direct Q(s,a)
            self.q_values = nn.Linear(hidden_size, action_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_size)
            
        Returns:
            q_values: (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.dueling:
            # Value stream
            v = F.relu(self.fc_value(x))
            v = self.value(v)  # (batch_size, 1)
            
            # Advantage stream
            a = F.relu(self.fc_advantage(x))
            a = self.advantage(a)  # (batch_size, action_size)
            
            # Dueling combination: Q = V + (A - mean(A))
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            return self.q_values(x)


class ActorNetwork(nn.Module):
    """
    Policy network for PPO (actor).
    
    Outputs: 
    - Mean of policy distribution (deterministic actions)
    - Log standard deviation (exploration)
    
    Why separate actor and critic?
    - Actor (policy) learns what actions to take
    - Critic (value) learns to evaluate state quality
    - This is more stable than learning both in one network
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Log standard deviation (learnable parameter for exploration)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute action means.
        
        Args:
            state: (batch_size, state_size)
            
        Returns:
            action_means: (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Softmax to ensure valid allocations (sum to 1)
        x = torch.softmax(x, dim=-1)
        return x
    
    def get_action_and_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> tuple:
        """
        Sample action and compute log probability.
        
        Args:
            state: (batch_size, state_size)
            action: Optional action to evaluate. If None, sample.
            
        Returns:
            action: (batch_size, action_size)
            log_prob: (batch_size,)
        """
        # Get action means (deterministic policy)
        action_means = self.forward(state)
        
        # Standard deviation (from learnable parameter)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_means, std)
        
        if action is None:
            # Sample action
            action = dist.rsample()
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to [0, 1] and normalize
        action = torch.clamp(action, 0, 1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Value network for PPO (critic).
    
    Outputs:
    - State value V(s): how good is this state?
    
    Why needed?
    - Provides baseline for advantage estimation
    - Stabilizes policy gradient updates
    - Reduces variance in gradient estimates
    """
    
    def __init__(
        self,
        state_size: int,
        hidden_size: int = 256
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            state: (batch_size, state_size)
            
        Returns:
            value: (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.value(x)
        return x


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone with separate actor and critic heads.
    
    This is more parameter-efficient than separate networks.
    Trade-off: shared representation vs. task-specific learning.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256
    ):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic head (value)
        self.critic_head = nn.Linear(hidden_size, 1)
    
    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_size)
            
        Returns:
            action_means: (batch_size, action_size)
            value: (batch_size, 1)
        """
        backbone_out = self.backbone(state)
        
        # Actor: policy
        action_logits = self.actor_head(backbone_out)
        action_means = torch.softmax(action_logits, dim=-1)
        
        # Critic: value
        value = self.critic_head(backbone_out)
        
        return action_means, value
    
    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> tuple:
        """
        Sample action, compute log prob, and get value estimate.
        
        Args:
            state: (batch_size, state_size)
            action: Optional action to evaluate
            
        Returns:
            action, log_prob, value
        """
        action_means, value = self.forward(state)
        
        # Stochastic policy
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_means, std)
        
        if action is None:
            action = dist.rsample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Normalize action
        action = torch.clamp(action, 0, 1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        return action, log_prob, value.squeeze(-1)


if __name__ == "__main__":
    # Test networks
    state_size = 20 * 5 + 5 + 1  # 106
    action_size = 5
    batch_size = 32
    
    # Test DQN
    dqn = DuelingQNetwork(state_size, 27)  # 27 discrete actions
    state = torch.randn(batch_size, state_size)
    q_values = dqn(state)
    print(f"DQN Q-values shape: {q_values.shape}")  # Should be (32, 27)
    
    # Test PPO networks
    actor = ActorNetwork(state_size, action_size)
    critic = CriticNetwork(state_size)
    
    action_means = actor(state)
    values = critic(state)
    
    print(f"Actor output shape: {action_means.shape}")  # Should be (32, 5)
    print(f"Critic output shape: {values.shape}")  # Should be (32, 1)
    print(f"Action means sum: {action_means.sum(dim=-1).mean().item():.4f}")  # Should be ~1.0
    
    # Test ActorCritic
    ac = ActorCriticNetwork(state_size, action_size)
    action, log_prob, value = ac.get_action_and_value(state)
    print(f"\nActorCritic:")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Value shape: {value.shape}")
