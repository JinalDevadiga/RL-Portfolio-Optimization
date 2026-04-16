"""
Synthetic Financial Data Generator

Creates realistic OHLCV data with:
- Correlated returns (like real assets)
- Time-varying volatility (heteroskedasticity)
- Trends and mean reversion
- Jump events (simulating news/events)

This is better for controlled RL experiments than using real data initially.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(
    n_days: int = 2000,
    n_assets: int = 5,
    starting_prices: np.ndarray = None,
    volatility: float = 0.015,
    correlation: float = 0.3,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic OHLCV data.
    
    Args:
        n_days: Number of trading days
        n_assets: Number of assets
        starting_prices: Starting close prices (default: 100 for all)
        volatility: Daily volatility (default: 1.5%)
        correlation: Asset correlation (default: 0.3)
        seed: Random seed
        
    Returns:
        prices: Shape (n_days, n_assets) array of close prices
    """
    np.random.seed(seed)
    
    if starting_prices is None:
        starting_prices = np.ones(n_assets) * 100
    
    # Create correlation matrix
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(corr_matrix)
    
    prices = starting_prices.copy()
    prices_history = [prices.copy()]
    
    # Time-varying volatility (GARCH-like)
    vol_history = [volatility] * n_assets
    vol_persistence = 0.7
    vol_mean_reversion = 0.015
    
    for day in range(n_days):
        # Generate correlated shocks
        uncorrelated_shocks = np.random.randn(n_assets)
        shocks = L @ uncorrelated_shocks
        
        # Update volatility (mean reversion)
        vol_history = [
            vol_persistence * v + (1 - vol_persistence) * vol_mean_reversion
            for v in vol_history
        ]
        
        # Add spikes (10% chance of significant event)
        if np.random.rand() < 0.10:
            shocks *= np.random.uniform(1.5, 2.5)
        
        # Apply shocks with time-varying volatility
        returns = shocks * np.array(vol_history)
        
        # Add drift (slight upward bias, like real markets)
        drift = 0.0002  # 0.02% daily drift
        returns += drift
        
        # Update prices (log returns)
        prices = prices * np.exp(returns)
        prices_history.append(prices.copy())
    
    return np.array(prices_history)


def create_realistic_data_with_regimes(
    n_days: int = 2000,
    n_assets: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Generate data with multiple market regimes (bull, bear, sideways).
    
    This is more realistic than pure geometric Brownian motion.
    """
    np.random.seed(seed)
    
    prices = np.ones((n_days, n_assets)) * 100
    
    # Define regimes
    regime_length = n_days // 4  # 4 regimes
    regimes = ['bull', 'bear', 'sideways', 'volatile']
    regime_params = {
        'bull': {'drift': 0.0008, 'vol': 0.010, 'corr': 0.5},
        'bear': {'drift': -0.0005, 'vol': 0.018, 'corr': 0.7},
        'sideways': {'drift': 0.0001, 'vol': 0.008, 'corr': 0.3},
        'volatile': {'drift': 0.0002, 'vol': 0.020, 'corr': 0.4},
    }
    
    # Correlation matrix
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_matrix[i, j] = 0.4
            corr_matrix[j, i] = 0.4
    
    L = np.linalg.cholesky(corr_matrix)
    
    for day in range(n_days):
        # Determine current regime
        regime_idx = day // regime_length
        regime = regimes[min(regime_idx, len(regimes) - 1)]
        params = regime_params[regime]
        
        # Update correlation matrix for current regime
        temp_corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                temp_corr[i, j] = params['corr']
                temp_corr[j, i] = params['corr']
        L = np.linalg.cholesky(temp_corr)
        
        # Generate shocks
        uncorrelated = np.random.randn(n_assets)
        shocks = L @ uncorrelated
        
        # Apply drift and volatility
        returns = shocks * params['vol'] + params['drift']
        
        # Update prices
        prices[day] = prices[day - 1] * np.exp(returns) if day > 0 else prices[0]
    
    return prices


def save_data(prices: np.ndarray, filename: str = 'data/data.csv'):
    """Save prices to CSV."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(
        prices,
        columns=[f'Asset_{i}' for i in range(prices.shape[1])]
    )
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(prices), freq='D')
    df = df[['Date'] + [f'Asset_{i}' for i in range(prices.shape[1])]]
    
    df.to_csv(filename, index=False)
    print(f"Saved {len(prices)} trading days to {filename}")
    
    return df


if __name__ == "__main__":
    # Generate and save data
    prices = generate_synthetic_data(n_days=2000, n_assets=5)
    df = save_data(prices)
    
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nPrice statistics:")
    print(df.describe())
