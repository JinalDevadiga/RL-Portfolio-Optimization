"""
Performance Metrics for Portfolio Analysis

Standard metrics used in finance and RL:
- Sharpe Ratio: Risk-adjusted return (most common)
- Sortino Ratio: Only penalizes downside volatility
- Calmar Ratio: Return vs. maximum drawdown
- Win Rate: Percentage of profitable days
- Maximum Drawdown: Worst-case loss

These give you a complete picture of strategy performance.
"""

import numpy as np
from typing import Dict


def compute_metrics(
    portfolio_values: np.ndarray,
    returns: np.ndarray = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Args:
        portfolio_values: Array of portfolio values over time
        returns: Daily returns (if None, computed from portfolio values)
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Trading days per year (default 252)
        
    Returns:
        Dictionary with all metrics
    """
    
    if len(portfolio_values) < 2:
        return {}
    
    # Compute returns from portfolio values if not provided
    if returns is None:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    returns = np.array(returns).flatten()
    values = np.array(portfolio_values).flatten()
    
    # Total return
    total_return = (values[-1] - values[0]) / values[0]
    
    # Annualized metrics
    num_periods = len(returns)
    years = num_periods / periods_per_year
    
    annual_return = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else 0
    annual_volatility = np.std(returns) * np.sqrt(periods_per_year)
    
    # Sharpe Ratio: (annual_return - rf) / annual_volatility
    sharpe_ratio = (annual_return - risk_free_rate) / (annual_volatility + 1e-8)
    
    # Sortino Ratio: only downside volatility
    downside_returns = returns[returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / (downside_volatility + 1e-8)
    
    # Maximum Drawdown
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    # Calmar Ratio: annual return / abs(max drawdown)
    calmar_ratio = annual_return / (abs(max_drawdown) + 1e-8)
    
    # Win Rate: percentage of positive return days
    win_rate = np.sum(returns > 0) / len(returns)
    
    # Profit Factor: sum of gains / sum of losses
    gains = np.sum(returns[returns > 0]) if np.any(returns > 0) else 0
    losses = abs(np.sum(returns[returns < 0])) if np.any(returns < 0) else 1
    profit_factor = gains / (losses + 1e-8)
    
    # Consecutive wins/losses
    num_trades = np.sum(returns != 0)
    
    # Volatility of returns
    return_vol = np.std(returns)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'daily_volatility': return_vol,
        'num_trades': num_trades,
        'final_value': values[-1],
        'initial_value': values[0]
    }


def compare_strategies(metrics_list: list, strategy_names: list) -> None:
    """
    Print comparison of multiple strategies.
    
    Args:
        metrics_list: List of metric dictionaries
        strategy_names: Names of strategies
    """
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    # Header
    header = f"{'Metric':<25}"
    for name in strategy_names:
        header += f"{name:>15}"
    print(header)
    print("-" * (25 + 15 * len(strategy_names)))
    
    # Metrics to display
    metric_keys = [
        ('total_return', 'Total Return', '.2%'),
        ('annual_return', 'Annual Return', '.2%'),
        ('annual_volatility', 'Annual Volatility', '.2%'),
        ('sharpe_ratio', 'Sharpe Ratio', '.4f'),
        ('sortino_ratio', 'Sortino Ratio', '.4f'),
        ('max_drawdown', 'Max Drawdown', '.2%'),
        ('calmar_ratio', 'Calmar Ratio', '.4f'),
        ('win_rate', 'Win Rate', '.1%'),
        ('profit_factor', 'Profit Factor', '.2f'),
    ]
    
    for key, display_name, fmt in metric_keys:
        row = f"{display_name:<25}"
        for metrics in metrics_list:
            value = metrics.get(key, 0)
            if fmt.endswith('f'):
                row += f"{value:{fmt}:>15}"
            else:
                row += f"{value:{fmt}:>15}"
        print(row)


def compute_rolling_metrics(
    portfolio_values: np.ndarray,
    window: int = 63,  # 3 months
    periods_per_year: int = 252
) -> Dict[str, np.ndarray]:
    """
    Compute rolling metrics (e.g., rolling Sharpe ratio).
    
    Args:
        portfolio_values: Portfolio values over time
        window: Rolling window size (in trading days)
        periods_per_year: Trading days per year
        
    Returns:
        Dictionary of rolling metrics
    """
    
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    
    rolling_sharpe = []
    rolling_volatility = []
    rolling_return = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        
        # Rolling Sharpe
        annual_return = window_returns.mean() * periods_per_year
        annual_vol = window_returns.std() * np.sqrt(periods_per_year)
        sharpe = annual_return / (annual_vol + 1e-8)
        rolling_sharpe.append(sharpe)
        
        # Rolling volatility
        rolling_volatility.append(annual_vol)
        
        # Rolling return
        rolling_return.append(annual_return)
    
    return {
        'sharpe': np.array(rolling_sharpe),
        'volatility': np.array(rolling_volatility),
        'return': np.array(rolling_return)
    }


if __name__ == "__main__":
    # Test metrics computation
    portfolio_values = np.exp(np.cumsum(np.random.randn(252) * 0.01))
    portfolio_values = portfolio_values * 100000  # Start at 100k
    
    metrics = compute_metrics(portfolio_values)
    
    print("Portfolio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test comparison
    metrics1 = compute_metrics(portfolio_values)
    metrics2 = compute_metrics(portfolio_values * 1.1)  # 10% higher values
    
    compare_strategies([metrics1, metrics2], ['Strategy A', 'Strategy B'])
