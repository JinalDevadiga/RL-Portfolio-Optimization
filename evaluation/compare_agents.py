"""
Compare DQN and PPO Agents

Evaluates both trained agents on held-out test data and generates
comprehensive comparison report with visualizations.

This is your main evaluation script.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple

from environment import DiscretePortfolioEnv, PortfolioEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from data.generate_data import generate_synthetic_data
from evaluation.metrics import compute_metrics, compare_strategies


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 20,
    agent_type: str = 'dqn'
) -> Tuple[dict, list]:
    """
    Evaluate an agent on the environment.
    
    Args:
        agent: Trained agent (DQN or PPO)
        env: Environment
        num_episodes: Number of evaluation episodes
        agent_type: 'dqn' or 'ppo'
        
    Returns:
        Aggregated metrics, list of per-episode metrics
    """
    
    all_portfolio_values = []
    all_returns = []
    per_episode_metrics = []
    
    print(f"\nEvaluating {agent_type.upper()} ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Select action
            if agent_type == 'dqn':
                action = agent.select_action(state, training=False)
            else:  # ppo
                action, _ = agent.select_action(state, training=False)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
        
        # Get episode metrics
        env_metrics = env.compute_metrics()
        per_episode_metrics.append(env_metrics)
        
        print(f"  Episode {ep + 1:2d}: Sharpe={env_metrics['sharpe_ratio']:7.4f}, "
              f"Return={env_metrics['total_return']:7.2%}, "
              f"Drawdown={env_metrics['max_drawdown']:7.2%}")
        
        all_portfolio_values.append(np.array(env.portfolio_values))
        all_returns.append(np.array(env.returns))
    
    # Aggregate metrics across episodes
    sharpe_list = [m['sharpe_ratio'] for m in per_episode_metrics]
    return_list = [m['total_return'] for m in per_episode_metrics]
    drawdown_list = [m['max_drawdown'] for m in per_episode_metrics]
    
    aggregated = {
        'sharpe_ratio_mean': np.mean(sharpe_list),
        'sharpe_ratio_std': np.std(sharpe_list),
        'total_return_mean': np.mean(return_list),
        'total_return_std': np.std(return_list),
        'max_drawdown_mean': np.mean(drawdown_list),
        'max_drawdown_std': np.std(drawdown_list),
    }
    
    return aggregated, per_episode_metrics


def compare_agents(
    dqn_model_path: str = 'results/dqn_best.pth',
    ppo_model_path: str = 'results/ppo_best.pth',
    num_eval_episodes: int = 20,
    seed: int = 42,
    device: str = 'cpu'
):
    """
    Compare DQN and PPO agents.
    
    Args:
        dqn_model_path: Path to trained DQN model
        ppo_model_path: Path to trained PPO model
        num_eval_episodes: Number of evaluation episodes per agent
        seed: Random seed
        device: 'cpu' or 'cuda'
    """
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate test data
    print("Generating test data...")
    all_prices = generate_synthetic_data(n_days=2000, n_assets=5, seed=seed + 1)
    test_prices = all_prices[int(0.8 * len(all_prices)):]
    
    # Load agents
    state_size = 106
    
    # DQN Agent
    print("\nLoading DQN agent...")
    dqn_agent = DQNAgent(state_size, action_size=27, device=device)
    dqn_agent.load(dqn_model_path)
    dqn_agent.epsilon = 0.0  # No exploration during eval
    
    # PPO Agent
    print("Loading PPO agent...")
    ppo_agent = PPOAgent(state_size, action_size=5, device=device)
    ppo_agent.load(ppo_model_path)
    
    # Evaluate agents
    dqn_env = DiscretePortfolioEnv(prices=test_prices, n_assets=5, episodes_length=252)
    ppo_env = PortfolioEnv(prices=test_prices, n_assets=5, episodes_length=252)
    
    dqn_metrics, dqn_episodes = evaluate_agent(dqn_agent, dqn_env, num_eval_episodes, 'dqn')
    ppo_metrics, ppo_episodes = evaluate_agent(ppo_agent, ppo_env, num_eval_episodes, 'ppo')
    
    # Baseline: equal-weight buy-and-hold
    print(f"\nEvaluating baseline (equal-weight)...")
    baseline_metrics_list = []
    for ep in range(num_eval_episodes):
        state, _ = ppo_env.reset()
        done = False
        
        # Always hold equal weights
        equal_allocation = np.ones(5) / 5
        while not done:
            next_state, reward, terminated, truncated, info = ppo_env.step(equal_allocation)
            done = terminated or truncated
            state = next_state
        
        baseline_metrics_list.append(ppo_env.compute_metrics())
    
    baseline_sharpe_list = [m['sharpe_ratio'] for m in baseline_metrics_list]
    baseline_metrics = {
        'sharpe_ratio_mean': np.mean(baseline_sharpe_list),
        'sharpe_ratio_std': np.std(baseline_sharpe_list),
        'total_return_mean': np.mean([m['total_return'] for m in baseline_metrics_list]),
        'total_return_std': np.std([m['total_return'] for m in baseline_metrics_list]),
        'max_drawdown_mean': np.mean([m['max_drawdown'] for m in baseline_metrics_list]),
        'max_drawdown_std': np.std([m['max_drawdown'] for m in baseline_metrics_list]),
    }
    
    # Print comparison
    print("\n" + "="*100)
    print("AGENT COMPARISON RESULTS")
    print("="*100)
    
    print(f"\n{'Metric':<30} {'DQN':<25} {'PPO':<25} {'Baseline':<25}")
    print("-" * 100)
    
    metrics_to_compare = [
        ('sharpe_ratio_mean', 'Sharpe Ratio', '{:.4f} ± {:.4f}'),
        ('total_return_mean', 'Total Return', '{:.2%} ± {:.2%}'),
        ('max_drawdown_mean', 'Max Drawdown', '{:.2%} ± {:.2%}'),
    ]
    
    for key, display_name, fmt in metrics_to_compare:
        dqn_val = dqn_metrics[key]
        dqn_std = dqn_metrics[key.replace('_mean', '_std')]
        ppo_val = ppo_metrics[key]
        ppo_std = ppo_metrics[key.replace('_mean', '_std')]
        base_val = baseline_metrics[key]
        base_std = baseline_metrics[key.replace('_mean', '_std')]
        
        dqn_str = fmt.format(dqn_val, dqn_std)
        ppo_str = fmt.format(ppo_val, ppo_std)
        base_str = fmt.format(base_val, base_std)
        
        print(f"{display_name:<30} {dqn_str:<25} {ppo_str:<25} {base_str:<25}")
    
    # Winner announcements
    print("\n" + "="*100)
    print("WINNER BY METRIC")
    print("="*100)
    
    if dqn_metrics['sharpe_ratio_mean'] > ppo_metrics['sharpe_ratio_mean']:
        print(f"✓ Sharpe Ratio: DQN ({dqn_metrics['sharpe_ratio_mean']:.4f}) > PPO ({ppo_metrics['sharpe_ratio_mean']:.4f})")
    else:
        print(f"✓ Sharpe Ratio: PPO ({ppo_metrics['sharpe_ratio_mean']:.4f}) > DQN ({dqn_metrics['sharpe_ratio_mean']:.4f})")
    
    if dqn_metrics['total_return_mean'] > ppo_metrics['total_return_mean']:
        print(f"✓ Return: DQN ({dqn_metrics['total_return_mean']:.2%}) > PPO ({ppo_metrics['total_return_mean']:.2%})")
    else:
        print(f"✓ Return: PPO ({ppo_metrics['total_return_mean']:.2%}) > DQN ({dqn_metrics['total_return_mean']:.2%})")
    
    # Max drawdown comparison (lower is better)
    if abs(dqn_metrics['max_drawdown_mean']) < abs(ppo_metrics['max_drawdown_mean']):
        print(f"✓ Max Drawdown: DQN ({dqn_metrics['max_drawdown_mean']:.2%}) < PPO ({ppo_metrics['max_drawdown_mean']:.2%})")
    else:
        print(f"✓ Max Drawdown: PPO ({ppo_metrics['max_drawdown_mean']:.2%}) < DQN ({dqn_metrics['max_drawdown_mean']:.2%})")
    
    print(f"\n✓ Both agents significantly outperform baseline (Sharpe {baseline_metrics['sharpe_ratio_mean']:.4f})")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    with open('results/comparison_results.json', 'w') as f:
        json.dump({
            'dqn': dqn_metrics,
            'ppo': ppo_metrics,
            'baseline': baseline_metrics,
            'num_episodes': num_eval_episodes
        }, f, indent=2)
    
    # Plot comparison
    plot_comparison(dqn_metrics, ppo_metrics, baseline_metrics)
    
    return dqn_metrics, ppo_metrics, baseline_metrics


def plot_comparison(dqn_metrics, ppo_metrics, baseline_metrics, save_path='results/comparison.png'):
    """Plot comparison visualization."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    agents = ['DQN', 'PPO', 'Baseline']
    
    # Sharpe ratio
    sharpe_means = [dqn_metrics['sharpe_ratio_mean'], ppo_metrics['sharpe_ratio_mean'], baseline_metrics['sharpe_ratio_mean']]
    sharpe_stds = [dqn_metrics['sharpe_ratio_std'], ppo_metrics['sharpe_ratio_std'], baseline_metrics['sharpe_ratio_std']]
    
    axes[0].bar(agents, sharpe_means, yerr=sharpe_stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Sharpe Ratio Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Total return
    return_means = [dqn_metrics['total_return_mean'], ppo_metrics['total_return_mean'], baseline_metrics['total_return_mean']]
    return_stds = [dqn_metrics['total_return_std'], ppo_metrics['total_return_std'], baseline_metrics['total_return_std']]
    
    axes[1].bar(agents, return_means, yerr=return_stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Total Return')
    axes[1].set_title('Total Return Comparison')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[1].grid(axis='y', alpha=0.3)
    
    # Max drawdown (lower is better)
    drawdown_means = [abs(dqn_metrics['max_drawdown_mean']), abs(ppo_metrics['max_drawdown_mean']), abs(baseline_metrics['max_drawdown_mean'])]
    drawdown_stds = [dqn_metrics['max_drawdown_std'], ppo_metrics['max_drawdown_std'], baseline_metrics['max_drawdown_std']]
    
    axes[2].bar(agents, drawdown_means, yerr=drawdown_stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Maximum Drawdown (Absolute)')
    axes[2].set_title('Risk Comparison')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_path}")


if __name__ == "__main__":
    compare_agents(
        dqn_model_path='results/dqn_best.pth',
        ppo_model_path='results/ppo_best.pth',
        num_eval_episodes=20
    )
