"""
Experience Replay Buffer

Stores transitions (s, a, r, s', done) and samples random minibatches for training.

Why replay buffer?
1. Decorrelates training signal (consecutive transitions are highly correlated)
2. Allows reuse of transitions (sample efficiency)
3. Enables prioritization (sample important transitions more often)

This is a standard deque-based implementation. For large-scale RL, you might use
PrioritizedReplayBuffer (sample by TD error) but standard replay is sufficient here.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    """
    Standard experience replay buffer for DQN.
    
    Stores transitions and allows random sampling for mini-batch learning.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch from the buffer.
        
        Args:
            batch_size: Size of minibatch to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            All are numpy arrays suitable for batching
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples to train."""
        return len(self.buffer) >= batch_size
    
    def __len__(self) -> int:
        """Current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay.
    
    Samples transitions with higher priority based on TD error (temporal difference error).
    This is more sample-efficient because we focus learning on surprising/difficult transitions.
    
    Trade-off: More computational overhead, but faster convergence.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Buffer size
            alpha: How much prioritization to use (0=no priority, 1=full priority)
            beta: Importance sampling correction (increases with training)
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Prioritization factor
        self.beta = beta    # Importance sampling correction
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = None
    ):
        """Add transition with priority based on TD error."""
        super().add(state, action, reward, next_state, done)
        
        # Use max priority for new experiences (encourage exploration)
        if td_error is None:
            priority = self.max_priority
        else:
            priority = abs(td_error) + 1e-6  # Small constant to avoid zero priority
        
        self.priorities.append(priority)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on new TD errors.
        
        Args:
            indices: Which samples were updated
            td_errors: New TD errors for those samples
        """
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):
                priority = abs(td_error) + 1e-6
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample with prioritization.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
            where weights are importance sampling corrections
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Compute sampling probabilities from priorities
        priorities = np.array(list(self.priorities))
        probs = np.power(priorities, self.alpha) / np.sum(np.power(priorities, self.alpha))
        
        # Sample indices according to priorities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = np.power(len(self.buffer) * probs[indices], -self.beta)
        weights = weights / weights.max()  # Normalize for stability
        
        # Get samples
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,  # Return indices for priority update
            weights.astype(np.float32)  # Importance sampling weights
        )


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(106)
        action = np.random.randint(0, 27)
        reward = np.random.randn()
        next_state = np.random.randn(106)
        done = np.random.rand() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
    
    # Sample a batch
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones = batch
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Batch states shape: {states.shape}")
    print(f"Batch actions shape: {actions.shape}")
    print(f"Batch rewards shape: {rewards.shape}")
    
    # Test prioritized buffer
    pbuffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(106)
        action = np.random.randint(0, 27)
        reward = np.random.randn()
        next_state = np.random.randn(106)
        done = np.random.rand() < 0.1
        td_error = np.random.rand()
        
        pbuffer.add(state, action, reward, next_state, done, td_error)
    
    # Sample with priorities
    batch = pbuffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones, indices, weights = batch
    
    print(f"\nPrioritized buffer size: {len(pbuffer)}")
    print(f"Batch states shape: {states.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
