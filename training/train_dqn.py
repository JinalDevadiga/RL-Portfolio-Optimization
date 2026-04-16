"""
DQN Training Loop

Trains a DQN agent on the portfolio environment.
Tracks performance metrics and saves checkpoints.

Key aspects:
1. Exploration-exploitation balance (epsilon decay)
2. Periodic evaluation on held-out data
3. Checkpointing best models
4. Comprehensive metrics tracking
"""

import numpy as np
import torch
from environment import DiscretePortfolioEnv
from agents.dqn_agent import DQNAgent
from data.generate_data import generate_synthetic_data
from evaluation.metrics import compute_metrics
import matplotlib.pyplot as plt
from pathlib import Path
import json


def train_dqn(
    num_episodes: int = 500,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    batch_size: int = 32,
    buffer_size: int = 100000,
    target_update_freq: int = 1000,
    eval_episodes: int = 10,
    eval_interval: int = 50,
    device: str = 'cpu',
    seed: int = 42
):
    """
    Train DQN agent.
    
    Args:
        num_episodes: Number of training episodes
        learning_rate: Agent learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration probability
        epsilon_decay: Decay per episode
        batch_size: Training batch size
        buffer_size: Replay buffer capacity
        target_update_freq: Steps between target network updates
        eval_episodes: Number of evaluation episodes
        eval_interval: Evaluate every N episodes
        device: 'cpu' or 'cuda'
        seed: Random seed
    """
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate training data (first 80%)
    print("Generating synthetic training data...")
    all_prices = generate_synthetic_data(n_days=2000, n_assets=5, seed=seed)
    train_prices = all_prices[:int(0.8 * len(all_prices))]
    eval_prices = all_prices[int(0.8 * len(all_prices)):]
    
    # Create environments
    train_env = DiscretePortfolioEnv(
        prices=train_prices,
        n_assets=5,
        initial_capital=100000,
        transaction_cost=0.001,
        episodes_length=252
    )
    
    eval_env = DiscretePortfolioEnv(
        prices=eval_prices,
        n_assets=5,
        initial_capital=100000,
        transaction_cost=0.001,
        episodes_length=252
    )
    
    # Create agent
    state_size = 20 * 5 + 5 + 1  # 106
    action_size = 27
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_frequency=target_update_freq,
        device=device
    )
    
    # Training loop
    print(f"\nTraining DQN for {num_episodes} episodes...")
    
    train_metrics = {
        'episode': [],
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': [],
        'avg_loss': []
    }
    
    eval_metrics = {
        'episode': [],
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': []
    }
    
    best_sharpe = -np.inf
    best_model_path = 'results/dqn_best.pth'
    Path('results').mkdir(exist_ok=True)
    
    for episode in range(num_episodes):
        state, _ = train_env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Compute training metrics
        metrics = train_env.compute_metrics()
        train_metrics['episode'].append(episode)
        train_metrics['total_return'].append(metrics.get('total_return', 0))
        train_metrics['sharpe_ratio'].append(metrics.get('sharpe_ratio', 0))
        train_metrics['max_drawdown'].append(metrics.get('max_drawdown', 0))
        train_metrics['avg_loss'].append(np.mean(episode_loss) if episode_loss else 0)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            print(f"\n--- Episode {episode + 1} ---")
            print(f"Training Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            
            # Evaluate on held-out data
            eval_sharpe_list = []
            eval_returns_list = []
            eval_drawdown_list = []
            
            for _ in range(eval_episodes):
                state, _ = eval_env.reset()
                done = False
                
                while not done:
                    action = agent.select_action(state, training=False)
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
            
            print(f"Eval Sharpe: {avg_eval_sharpe:.4f} (μ={np.mean(eval_sharpe_list):.4f}, σ={np.std(eval_sharpe_list):.4f})")
            print(f"Eval Return: {avg_eval_return:.4f}")
            print(f"Eval Drawdown: {avg_eval_drawdown:.4f}")
            
            # Save best model
            if avg_eval_sharpe > best_sharpe:
                best_sharpe = avg_eval_sharpe
                agent.save(best_model_path)
                print(f"✓ Saved best model (Sharpe: {best_sharpe:.4f})")
    
    # Save final model
    agent.save('results/dqn_final.pth')
    
    # Save metrics
    with open('results/dqn_metrics.json', 'w') as f:
        json.dump({
            'train': train_metrics,
            'eval': eval_metrics
        }, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"Best eval Sharpe: {best_sharpe:.4f}")
    print(f"Models saved to results/")
    
    return agent, train_metrics, eval_metrics


def plot_training(train_metrics, eval_metrics, save_path='results/dqn_training.png'):
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
    
    # Loss
    axes[1, 1].plot(train_metrics['episode'], train_metrics['avg_loss'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Loss')
    axes[1, 1].set_title('Training Loss Over Time')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    # Train
    agent, train_metrics, eval_metrics = train_dqn(
        num_episodes=500,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_decay=0.995,
        batch_size=32,
        eval_interval=50,
        device='cpu'
    )
    
    # Plot
    plot_training(train_metrics, eval_metrics)
