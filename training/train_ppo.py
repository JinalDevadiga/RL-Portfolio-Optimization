"""
PPO Training Loop

Trains a PPO agent on the portfolio environment.
Much faster than DQN because it's on-policy and doesn't need replay buffer.
"""

import numpy as np
import torch
from environment import PortfolioEnv
from agents.ppo_agent import PPOAgent
from data.generate_data import generate_synthetic_data
import matplotlib.pyplot as plt
from pathlib import Path
import json


def train_ppo(
    num_episodes: int = 100,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    epochs: int = 10,
    batch_size: int = 64,
    entropy_coef: float = 0.01,
    eval_episodes: int = 10,
    eval_interval: int = 10,
    device: str = 'cpu',
    seed: int = 42
):
    """
    Train PPO agent.
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        epochs: PPO update epochs per batch
        batch_size: Training batch size
        entropy_coef: Entropy coefficient for exploration
        eval_episodes: Number of eval episodes
        eval_interval: Evaluate every N episodes
        device: 'cpu' or 'cuda'
        seed: Random seed
    """
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate training data
    print("Generating synthetic training data...")
    all_prices = generate_synthetic_data(n_days=2000, n_assets=5, seed=seed)
    train_prices = all_prices[:int(0.8 * len(all_prices))]
    eval_prices = all_prices[int(0.8 * len(all_prices)):]
    
    # Create environments
    train_env = PortfolioEnv(
        prices=train_prices,
        n_assets=5,
        initial_capital=100000,
        transaction_cost=0.001,
        episodes_length=252
    )
    
    eval_env = PortfolioEnv(
        prices=eval_prices,
        n_assets=5,
        initial_capital=100000,
        transaction_cost=0.001,
        episodes_length=252
    )
    
    # Create agent
    state_size = 20 * 5 + 5 + 1  # 106
    action_size = 5
    
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        epochs=epochs,
        batch_size=batch_size,
        entropy_coef=entropy_coef,
        device=device
    )
    
    print(f"\nTraining PPO for {num_episodes} episodes...")
    
    train_metrics = {
        'episode': [],
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': [],
        'policy_loss': [],
        'value_loss': []
    }
    
    eval_metrics = {
        'episode': [],
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': []
    }
    
    best_sharpe = -np.inf
    best_model_path = 'results/ppo_best.pth'
    Path('results').mkdir(exist_ok=True)
    
    for episode in range(num_episodes):
        state, _ = train_env.reset()
        done = False
        
        # Collect trajectory
        while not done:
            # Select action
            action, log_prob = agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Estimate value
            value = agent.estimate_value(state)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
        
        # Update policy and value function
        policy_loss, value_loss = agent.update()
        
        # Compute training metrics
        metrics = train_env.compute_metrics()
        train_metrics['episode'].append(episode)
        train_metrics['total_return'].append(metrics.get('total_return', 0))
        train_metrics['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
        train_metrics['max_drawdown'].append(metrics.get('max_drawdown', 0))
        train_metrics['policy_loss'].append(policy_loss if policy_loss is not None else 0)
        train_metrics['value_loss'].append(value_loss if value_loss is not None else 0)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\n--- Episode {episode + 1} ---")
            print(f"Training Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"Policy Loss: {policy_loss:.6f}, Value Loss: {value_loss:.6f}")
            
            # Evaluate
            eval_sharpe_list = []
            eval_returns_list = []
            eval_drawdown_list = []
            
            for _ in range(eval_episodes):
                state, _ = eval_env.reset()
                done = False
                
                while not done:
                    action, _ = agent.select_action(state, training=False)
                    next_state, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    state = next_state
                
                eval_met = eval_env.compute_metrics()
                eval_sharpe_list.append(eval_met.get('sharpe_ratio', 0))
                eval_returns_list.append(eval_met.get('total_return', 0))
                eval_drawdown_list.append(eval_met.get('max_drawdown', 0))
            
            avg_eval_sharpe = np.mean(eval_sharpe_list)
            avg_eval_return = np.mean(eval_returns_list)
            avg_eval_drawdown = np.mean(eval_drawdown_list)
            
            eval_metrics['episode'].append(episode)
            eval_metrics['total_return'].append(avg_eval_return)
            eval_metrics['sharpe_ratio'].append(avg_eval_sharpe)
            eval_metrics['max_drawdown'].append(avg_eval_drawdown)
            
            print(f"Eval Sharpe: {avg_eval_sharpe:.4f}")
            print(f"Eval Return: {avg_eval_return:.4f}")
            print(f"Eval Drawdown: {avg_eval_drawdown:.4f}")
            
            # Save best model
            if avg_eval_sharpe > best_sharpe:
                best_sharpe = avg_eval_sharpe
                agent.save(best_model_path)
                print(f"✓ Saved best model (Sharpe: {best_sharpe:.4f})")
    
    # Save final model
    agent.save('results/ppo_final.pth')
    
    # Save metrics
    with open('results/ppo_metrics.json', 'w') as f:
        json.dump({
            'train': train_metrics,
            'eval': eval_metrics
        }, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Best eval Sharpe: {best_sharpe:.4f}")
    
    return agent, train_metrics, eval_metrics


def plot_training(train_metrics, eval_metrics, save_path='results/ppo_training.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Sharpe ratio
    axes[0, 0].plot(train_metrics['episode'], train_metrics['sharpe_ratio'], label='Train', alpha=0.7)
    axes[0, 0].plot(eval_metrics['episode'], eval_metrics['sharpe_ratio'], label='Eval', marker='o')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('Sharpe Ratio Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Total return
    axes[0, 1].plot(train_metrics['episode'], train_metrics['total_return'], label='Train', alpha=0.7)
    axes[0, 1].plot(eval_metrics['episode'], eval_metrics['total_return'], label='Eval', marker='o')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Return')
    axes[0, 1].set_title('Total Return Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Max drawdown
    axes[1, 0].plot(train_metrics['episode'], train_metrics['max_drawdown'], label='Train', alpha=0.7)
    axes[1, 0].plot(eval_metrics['episode'], eval_metrics['max_drawdown'], label='Eval', marker='o')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Max Drawdown')
    axes[1, 0].set_title('Maximum Drawdown Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Losses
    axes[1, 1].plot(train_metrics['episode'], train_metrics['policy_loss'], label='Policy', alpha=0.7)
    axes[1, 1].plot(train_metrics['episode'], train_metrics['value_loss'], label='Value', alpha=0.7)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Loss Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    # Train
    agent, train_metrics, eval_metrics = train_ppo(
        num_episodes=100,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        eval_interval=10,
        device='cpu'
    )
    
    # Plot
    plot_training(train_metrics, eval_metrics)
