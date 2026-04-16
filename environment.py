"""
Portfolio Optimization Environment

A custom Gym environment for RL agents to learn portfolio allocation.
States: price history + current allocation + market volatility
Actions: target portfolio allocations
Rewards: Sharpe ratio + transaction cost penalty

This environment is designed to be realistic: it includes transaction costs,
enforces allocation constraints, and provides rich market information.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class PortfolioEnv(gym.Env):
    """
    Portfolio allocation environment.
    
    State: 
        - Last 20 days of returns for each asset (20, n_assets)
        - Current allocation (n_assets,)
        - Portfolio volatility estimate (scalar)
        
    Action (Continuous):
        - Target allocation for each asset in [0, 1], summing to 1
        
    Reward:
        - Daily Sharpe ratio improvement - transaction cost penalty
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self, 
        prices: np.ndarray,
        n_assets: int = 5,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% per transaction
        history_length: int = 20,
        episodes_length: int = 252,  # 1 trading year
        seed: int = 42
    ):
        """
        Args:
            prices: Shape (T, n_assets) - daily close prices
            n_assets: Number of assets in portfolio
            initial_capital: Starting portfolio value
            transaction_cost: Cost per transaction (e.g., 0.001 = 0.1%)
            history_length: Number of past days to include in state
            episodes_length: Days per episode
            seed: Random seed
        """
        super().__init__()
        
        self.prices = prices
        self.n_assets = n_assets
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.history_length = history_length
        self.episode_length = episodes_length
        
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # State space: [history (20, n_assets) + allocation (n_assets,) + volatility (1,)]
        # Flattened: 20*n_assets + n_assets + 1 = 21*n_assets + 1
        state_size = history_length * n_assets + n_assets + 1
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_size,), 
            dtype=np.float32
        )
        
        # Action space: continuous allocation (n_assets,) in [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(n_assets,), 
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.current_step = 0
        self.start_idx = 0
        self.portfolio_value = initial_capital
        self.allocation = np.ones(n_assets) / n_assets  # Equal weight
        self.held_shares = None  # Will be set in reset()
        
        # History tracking
        self.portfolio_values = []
        self.allocations_history = []
        self.returns = []
        
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        # Randomly select start date (leave enough data for history + episode)
        max_start = len(self.prices) - self.history_length - self.episode_length
        self.start_idx = self.rng.randint(0, max_start)
        self.current_step = 0
        
        # Reset portfolio
        self.portfolio_value = self.initial_capital
        self.allocation = np.ones(self.n_assets) / self.n_assets
        
        # Calculate initial shares held
        prices_today = self.prices[self.start_idx + self.history_length]
        capital_per_asset = self.portfolio_value * self.allocation
        self.held_shares = capital_per_asset / prices_today
        
        # Reset history
        self.portfolio_values = [self.portfolio_value]
        self.allocations_history = [self.allocation.copy()]
        self.returns = []
        
        return self._get_state(), {}
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        price_idx = self.start_idx + self.history_length + self.current_step
        
        # Get price history (last 20 days)
        history_start = price_idx - self.history_length
        history_end = price_idx
        price_history = self.prices[history_start:history_end]  # Shape: (20, n_assets)
        
        # Normalize by today's price (relative returns)
        today_prices = self.prices[price_idx]
        normalized_history = (price_history - today_prices) / today_prices
        
        # Compute volatility (realized vol over last 20 days)
        returns_20d = np.diff(price_history, axis=0) / price_history[:-1]
        volatility = np.std(returns_20d, axis=0).mean()
        volatility = np.clip(volatility, 0.001, 1.0)  # Avoid extremes
        
        # Construct state: [flattened history, allocation, volatility]
        state = np.concatenate([
            normalized_history.flatten(),      # (20 * n_assets,)
            self.allocation,                   # (n_assets,)
            [volatility]                       # (1,)
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Target allocation (n_assets,), will be normalized to sum to 1
            
        Returns:
            state, reward, terminated, truncated, info
        """
        price_idx = self.start_idx + self.history_length + self.current_step
        today_prices = self.prices[price_idx]
        
        # Normalize action to valid allocation
        target_allocation = action / (action.sum() + 1e-8)
        target_allocation = np.clip(target_allocation, 0, 1)
        target_allocation = target_allocation / (target_allocation.sum() + 1e-8)
        
        # === OLD PRICES (BEFORE MARKET MOVES) ===
        old_portfolio_value = self.portfolio_value
        old_prices = self.prices[price_idx]
        
        # === REBALANCE (trade to new allocation) ===
        # Current holdings value at old prices
        current_values = self.held_shares * old_prices
        old_allocations = current_values / current_values.sum()
        
        # Cost of rebalancing
        rebalance_vector = target_allocation - old_allocations
        rebalance_cost = np.sum(np.abs(rebalance_vector)) * self.portfolio_value * self.transaction_cost
        
        # Update portfolio value (after transaction costs)
        self.portfolio_value -= rebalance_cost
        
        # Calculate new shares after rebalancing
        capital_per_asset = self.portfolio_value * target_allocation
        self.held_shares = capital_per_asset / old_prices
        self.allocation = target_allocation.copy()
        
        # === NEW PRICES (NEXT DAY) ===
        if self.current_step < self.episode_length - 1:
            next_price_idx = price_idx + 1
            next_prices = self.prices[next_price_idx]
        else:
            # Last day of episode: use same prices
            next_prices = old_prices
        
        # Mark-to-market portfolio value
        portfolio_value_new = np.sum(self.held_shares * next_prices)
        
        # Daily return
        daily_return = (portfolio_value_new - old_portfolio_value) / old_portfolio_value
        
        # === REWARD: Sharpe Ratio Proxy + Transaction Cost Penalty ===
        # We approximate Sharpe ratio with recent returns
        self.returns.append(daily_return)
        
        if len(self.returns) >= 20:
            recent_returns = np.array(self.returns[-20:])
            mean_return = recent_returns.mean()
            std_return = recent_returns.std() + 1e-8
            sharpe_approx = mean_return / std_return * np.sqrt(252)  # Annualize
        else:
            sharpe_approx = 0.0
        
        # Penalty for transaction costs (already paid, but penalize in reward)
        transaction_penalty = rebalance_cost / self.portfolio_value * 0.1
        
        # Reward = Sharpe ratio - transaction penalty
        reward = sharpe_approx - transaction_penalty
        
        # Clip reward for stability
        reward = np.clip(reward, -2, 2)
        
        # Update state
        self.portfolio_value = portfolio_value_new
        self.portfolio_values.append(self.portfolio_value)
        self.allocations_history.append(self.allocation.copy())
        self.current_step += 1
        
        # Termination check
        terminated = self.current_step >= self.episode_length
        truncated = self.portfolio_value <= 0  # Bankruptcy
        
        next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'daily_return': daily_return,
            'allocation': self.allocation.copy(),
            'transaction_cost': rebalance_cost,
            'sharpe_proxy': sharpe_approx
        }
        
        return next_state, float(reward), terminated, truncated, info
    
    def compute_metrics(self) -> Dict:
        """
        Compute performance metrics over the episode.
        
        Returns:
            Dict with Sharpe ratio, max drawdown, cumulative return, etc.
        """
        values = np.array(self.portfolio_values)
        returns = np.array(self.returns)
        
        if len(values) < 2:
            return {}
        
        # Cumulative return
        total_return = (values[-1] - values[0]) / values[0]
        
        # Sharpe ratio (assuming 0% risk-free rate)
        annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (annual_volatility + 1e-8)
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / (downside_volatility + 1e-8)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': values[-1],
        }


class DiscretePortfolioEnv(PortfolioEnv):
    """
    Discrete version of portfolio environment for DQN.
    
    Action space: 27 discrete actions representing different rebalancing strategies.
    This is useful for interpretability and when you want clear buy/hold/sell signals.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Map discrete actions to allocations
        # 27 actions from all combinations of [0%, 33%, 67%, 100%] for 3 assets
        # (Simplified to 3 assets for computational tractability)
        self.action_allocations = self._create_action_space()
        self.action_space = spaces.Discrete(len(self.action_allocations))
    
    def _create_action_space(self) -> np.ndarray:
        """
        Create discrete action space as allocation vectors.
        Uses a simplified 3-asset space for interpretability.
        """
        allocations = []
        
        # Generate all combinations of 3 assets with 4 levels each: 0%, 33%, 67%, 100%
        levels = [0.0, 0.33, 0.67, 1.0]
        
        for a1 in levels:
            for a2 in levels:
                for a3 in levels:
                    # Normalize to sum to 1
                    alloc = np.array([a1, a2, a3])
                    total = alloc.sum()
                    if total > 0:
                        alloc = alloc / total
                    allocations.append(alloc)
        
        return np.array(allocations)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step with discrete action.
        
        Args:
            action: Integer action index (0 to 26)
        """
        # Convert discrete action to continuous allocation
        continuous_action = self.action_allocations[action].copy()
        
        # Pad to n_assets (if environment has more than 3 assets)
        if self.n_assets > 3:
            padded_action = np.zeros(self.n_assets)
            padded_action[:3] = continuous_action
            # Distribute remaining to other assets
            remaining = 1 - continuous_action.sum()
            for i in range(3, self.n_assets):
                padded_action[i] = remaining / (self.n_assets - 3)
            continuous_action = padded_action
        
        # Use parent class step function
        return super().step(continuous_action)


if __name__ == "__main__":
    # Quick test
    from data.generate_data import generate_synthetic_data
    
    prices = generate_synthetic_data(n_days=2000, n_assets=5)
    env = PortfolioEnv(prices, n_assets=5)
    
    state, _ = env.reset()
    print(f"State shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    
    # Random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward:.6f}")
    print(f"Portfolio value: {info['portfolio_value']:.2f}")
    print(f"Allocation: {info['allocation']}")
